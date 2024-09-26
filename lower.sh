#!/usr/bin/env bash
# Remember the current working directory
cwd=$(pwd)

# heir should be run using bazel from the root of the heir repo
heir_dir=/home/ubuntu/heir
cd $heir_dir

# Set the directory containing the MLIR files
mlir_dir=/home/ubuntu/simplemlir

# Function to print step name with or without figlet
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

# Function to run heir-opt command
run_heir_opt() {
    local step_name="$1"
    local options="$2"
    local input_num="$3"
    local output_num="$4"

    print_step_name "$step_name"
    {
        bazel run //tools:heir-opt -- $options ${mlir_dir}/${input_num}.mlir > ${mlir_dir}/${output_num}.mlir
    } || {
        echo "ERROR ($step_name): failed to heir-opt $options"
        exit 2
    }
}

# Function to run heir-translate command
run_heir_translate() {
    local step_name="$1"
    local options="$2"
    local input_num="$3"
    local output_num="$4"
    local extension="$5"

    print_step_name "$step_name"
    {
        bazel run //tools:heir-translate -- $options ${mlir_dir}/${input_num}.mlir > ${mlir_dir}/${output_num}.${extension}
    } || {
        echo "ERROR ($step_name): failed to heir-translate $options"
        exit 2
    }
}

# example run_heir_opt call:
# run_heir_opt "canonicalize" \
#   "--canonicalize" \
#   "in.mlir" \
#   "out.mlir"

# 1) Convert TOSA to Arith
run_heir_opt "tosa 2 arith" \
  "--heir-tosa-to-arith" \
  "00_tosa.matmul" \
  "01_matmul"

run_heir_opt "canonicalize" \
  "--canonicalize" \
  "01_matmul" \
  "02_canonicalize"

run_heir_opt "secretize" \
  "--secretize --wrap-generic" \
  "02_canonicalize" \
  "03_secretize"

run_heir_opt "mlir 2 openfhe" \
  "--secret-to-bgv" \
  "03_secretize" \
  "04_mlir_openfhe"

# 4) Translate to OpenFHE C++ code
run_heir_translate "emit openfhe" \
  "--emit-openfhe-pke" \
  "04_mlir_openfhe" \
  "05_emit_openfhe" \
  "C"
run_heir_translate "emit openfhe header" \
  "--emit-openfhe-pke-header" \
  "04_mlir_openfhe" \
  "04_emit_openfhe" \
  "h"

cd $cwd
