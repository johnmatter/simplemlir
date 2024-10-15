import mlir
from mlir.ir import Context, Module, InsertionPoint, Location, FunctionType, RankedTensorType, UnrankedTensorType
from mlir.dialects import func, tosa
import pdb

def find_function_location(module: Module, ctx: Context, function_name: str) -> Location:
    """
    Find the Location of a named function in the given module.
    
    Args:
    module (Module): The MLIR module to search in.
    ctx (Context): The MLIR context.
    function_name (str): The name of the function to find.
    
    Returns:
    Location: The Location of the found function.
    
    Raises:
    ValueError: If the function is not found in the module.
    """
    with ctx:
        for op in module.body:
            if isinstance(op, func.FuncOp) and op.name.value == function_name:
                return op.location
    raise ValueError(f"Function '{function_name}' not found in the module.")

def wrap_and_flatten_function(module: Module, ctx: Context, function_name: str, loc: Location) -> Module:
    """
    Wrap the specified function at the given Location and flatten its arguments.
    
    Args:
    module (Module): The original MLIR module.
    ctx (Context): The MLIR context.
    function_name (str): The name of the function to wrap and flatten.
    location (Location): The Location of the function to wrap.
    
    Returns:
    Module: A new module with the wrapped and flattened function.
    """
    with ctx, loc:
        new_module = Module.create()
        with InsertionPoint(new_module.body):
            for op in module.body:
                if isinstance(op, func.FuncOp) and op.name.value == function_name:
                    # Get the original function type
                    orig_type = op.type

                    # Flatten input tensors
                    flattened_input_types = []
                    for input_type in orig_type.inputs:
                        if isinstance(input_type, RankedTensorType):
                            flattened_shape = [1, input_type.shape[0] * input_type.shape[1]]
                            flattened_input_type = RankedTensorType.get(flattened_shape, input_type.element_type)
                            flattened_input_types.append(flattened_input_type)
                        else:
                            flattened_input_types.append(input_type)

                    # Create a new function type with flattened inputs
                    new_func_type = FunctionType.get(flattened_input_types, orig_type.results)

                    # Create a new function with the updated type
                    new_func = func.FuncOp(op.name, new_func_type, loc=loc)
                    
                    # Create a new block for the function body
                    entry_block = new_func.add_entry_block()
                    
                    # Copy the function body
                    with InsertionPoint(entry_block):
                        # Add reshaping operations for flattened inputs
                        reshaped_args = []
                        for i, (orig_input, flattened_input) in enumerate(zip(orig_type.inputs, flattened_input_types)):
                            if isinstance(orig_input, RankedTensorType) and isinstance(flattened_input, RankedTensorType):
                                reshaped_input = tosa.reshape(new_func.arguments[i], orig_input.shape)
                                reshaped_args.append(reshaped_input)
                            else:
                                reshaped_args.append(new_func.arguments[i])
                        
                        breakpoint()
                        # Clone the original function body, replacing arguments
                        for old_op in op.body.blocks[0].operations:
                            new_op = old_op.clone()
                            for i, old_arg in enumerate(op.arguments):
                                for operand in new_op.operands:
                                    if operand == old_arg:
                                        operand.replace_all_uses_with(reshaped_args[i])
                            if entry_block.terminator:
                                new_op.operation.move_before(entry_block.terminator)
                            else:
                                entry_block.append(new_op)

                else:
                    # Copy non-function operations as-is
                    op.operation.clone()

    return new_module

if __name__ == "__main__":
    try:
        with Context() as ctx:
            module = Module.parse('''
            func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
                %lhs_3D = tosa.reshape %arg0 {new_shape = array<i64 : 1, 2, 2>} : (tensor<2x2xi32>) -> tensor<1x2x2xi32>
                %rhs_3D = tosa.reshape %arg1 {new_shape = array<i64 : 1, 2, 2>} : (tensor<2x2xi32>) -> tensor<1x2x2xi32>
                %2 = tosa.matmul %lhs_3D, %rhs_3D : (tensor<1x2x2xi32>, tensor<1x2x2xi32>) -> tensor<1x2x2xi32>
                %3 = tosa.reshape %2 {new_shape = array<i64 : 2, 2>}  : (tensor<1x2x2xi32>) -> tensor<2x2xi32>
                return %3 : tensor<2x2xi32>
            }
            ''', ctx)

            function_name = "main"
            function_location = find_function_location(module, ctx, function_name)
            wrapped_module = wrap_and_flatten_function(module, ctx, function_name, function_location)

            print(wrapped_module)

    except Exception as e:
        print(f"An error occurred: {e}")