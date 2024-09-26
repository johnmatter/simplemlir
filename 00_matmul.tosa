func.func @main(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %lhs_3D = tosa.reshape %arg0 {new_shape = array<i64 : 1, 2, 2>} : (tensor<2x2xi32>) -> tensor<1x2x2xi32>
    %rhs_3D = tosa.reshape %arg1 {new_shape = array<i64 : 1, 2, 2>} : (tensor<2x2xi32>) -> tensor<1x2x2xi32>
    %2 = tosa.matmul %lhs_3D, %rhs_3D : (tensor<1x2x2xi32>, tensor<1x2x2xi32>) -> tensor<1x2x2xi32>
    
    // Extract the result as a 1x1 tensor (scalar)
    // %3 = tosa.slice %2 {start = array<i64: 0, 0>, size = array<i64: 1, 1>} : (tensor<2x2xi32>) -> tensor<1x1xi32>
    
    %3 = tosa.reshape %2 {new_shape = array<i64 : 2, 2>}  : (tensor<1x2x2xi32>) -> tensor<2x2xi32>
    return %3 : tensor<2x2xi32>
}
