func.func @main(%arg0: tensor<4xi32>, %arg1: tensor<4xi32>) -> tensor<4xi32> {
    %0 = tosa.reshape %arg0 {new_shape = array<i64: 2, 2>} : (tensor<4xi32>) -> tensor<2x2xi32>
    %1 = tosa.reshape %arg1 {new_shape = array<i64: 2, 2>} : (tensor<4xi32>) -> tensor<2x2xi32>
    %lhs_3D = tosa.reshape %0 {new_shape = array<i64 : 1, 2, 2>} : (tensor<2x2xi32>) -> tensor<1x2x2xi32>
    %rhs_3D = tosa.reshape %1 {new_shape = array<i64 : 1, 2, 2>} : (tensor<2x2xi32>) -> tensor<1x2x2xi32>
    %mul_out = tosa.matmul %lhs_3D, %rhs_3D : (tensor<1x2x2xi32>, tensor<1x2x2xi32>) -> tensor<1x2x2xi32>
    %mul_out_2d = tosa.reshape %mul_out {new_shape = array<i64 : 2, 2>}  : (tensor<1x2x2xi32>) -> tensor<2x2xi32>
    %mul_out_flat = tosa.reshape %mul_out_2d {new_shape = array<i64 : 4>} : (tensor<2x2xi32>) -> tensor<4xi32>
    return %mul_out_flat : tensor<4xi32>
}
