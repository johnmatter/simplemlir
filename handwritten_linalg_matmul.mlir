module {
  func.func @main(%vec : !secret.secret<tensor<1x4xf16>>) -> !secret.secret<tensor<1x4xf16>> {
    %matrix = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]> : tensor<4x4xf16>
    %bias = arith.constant dense<[[17.0, 18.0, 19.0, 20.0]]> : tensor<1x4xf16>
    %out = secret.generic ins (%vec : !secret.secret<tensor<1x4xf16>>) {
    ^bb0(%converted_vec: tensor<1x4xf16>):
      %0 = linalg.matmul ins(%converted_vec, %matrix : tensor<1x4xf16>, tensor<4x4xf16>) outs(%bias : tensor<1x4xf16>) -> tensor<1x4xf16>
      secret.yield %0 : tensor<1x4xf16>
    } -> !secret.secret<tensor<1x4xf16>>
    return %out : !secret.secret<tensor<1x4xf16>>
  }
}