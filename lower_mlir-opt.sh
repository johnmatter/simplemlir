#!/usr/bin/env bash
# Remember the current working directory
cwd=$(pwd)

# heir should be run using bazel from the root of the heir repo
heir_dir=/home/ubuntu/heir
cd $heir_dir

# Set the directory containing the MLIR files
mlir_dir=/home/ubuntu/simplemlir

# Function to print a "header" (with or without figlet) to stdout
print_step_name() {
    local step_name="$1"
    if command -v figlet &> /dev/null; then
        figlet "$step_name"
    else
        local line=$(printf "%0.s-" $(seq 1 ${#step_name}))
        echo
        echo "$line"
        echo "$step_name"
        echo "$line"
        echo
    fi
}

print_step_name "mlir-opt"
print_step_name "lower tosa"
mlir-opt \
  --pass-pipeline='builtin.module(func.func(tosa-to-linalg-named, tosa-to-tensor, convert-linalg-to-affine-loops))' \
  $mlir_dir/00_matmul.tosa \
  > $mlir_dir/01_tensor_linalg.mlir

print_step_name "heir-opt secretize"
bazel run //tools:heir-opt -- \
  --secretize \
  --wrap-generic \
  $mlir_dir/01_tensor_linalg.mlir \
  > $mlir_dir/02_secretize.mlir \
  2> $mlir_dir/02.err

print_step_name "heir-opt to openfhe-bgv"
degree=4
bazel run //tools:heir-opt -- \
  --mlir-to-openfhe-bgv="entry-function=main ciphertext-degree=$degree" \
  $mlir_dir/02_secretize.mlir \
  > $mlir_dir/03_openfhe.mlir \
  2> $mlir_dir/03.err
