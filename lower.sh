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

# Function to run heir command (opt or translate)
#
# This function executes a specified heir command (either 'opt' or 'translate')
# with given options on an input file and saves the output to a specified output
# file.
#
# Arguments:
#   $1 (step_name): A string describing the current step (used for logging)
#   $2 (command): The heir command to run ('opt' or 'translate')
#   $3 (options): Command-line options to pass to the heir command
#   $4 (input_file): The name of the input file (relative to $mlir_dir)
#   $5 (output_file): The name of the output file (relative to $mlir_dir)
#
# Output:
#   Writes the command output to the specified output file.
#   Prints step name using print_step_name function.
#   In case of error, prints an error message and exits with status code 2.
run_heir() {
    local step_name="$1"
    local command="$2"
    local options="$3"
    local input_file="$4"
    local output_file="$5"

    print_step_name "$step_name"
    {
        bazel run //tools:heir-$command -- \
          $options \
          ${mlir_dir}/${input_file} \
          > ${mlir_dir}/${output_file}
    } || {
        echo "ERROR ($step_name): failed to run heir-$command $options"
        exit 2
    }
}

#  _          _                       _   
# | |__   ___(_)_ __       ___  _ __ | |_ 
# | '_ \ / _ \ | '__|____ / _ \| '_ \| __|
# | | | |  __/ | | |_____| (_) | |_) | |_ 
# |_| |_|\___|_|_|        \___/| .__/ \__|
#                              |_|

run_heir "to arith" \
  "opt" \
  "--heir-tosa-to-arith" \
  "00_matmul.tosa" \
  "01_matmul.mlir"

run_heir "canonicalize" \
  "opt" \
  "--canonicalize" \
  "01_matmul.mlir" \
  "02_canonicalize.mlir"

run_heir "secretize" \
  "opt" \
  "--secretize --wrap-generic" \
  "02_canonicalize.mlir" \
  "03_secretize.mlir"

run_heir "to openfhe" \
  "opt" \
  "--secret-to-bgv" \
  "03_secretize.mlir" \
  "04_mlir_openfhe.mlir"

#  _          _           _                       _       _       
# | |__   ___(_)_ __     | |_ _ __ __ _ _ __  ___| | __ _| |_ ___ 
# | '_ \ / _ \ | '__|____| __| '__/ _` | '_ \/ __| |/ _` | __/ _ \
# | | | |  __/ | | |_____| |_| | | (_| | | | \__ \ | (_| | ||  __/
# |_| |_|\___|_|_|        \__|_|  \__,_|_| |_|___/_|\__,_|\__\___|

run_heir "emit openfhe" \
  "translate" \
  "--emit-openfhe-pke" \
  "04_mlir_openfhe.mlir" \
  "05_emit_openfhe.cpp" \

run_heir "emit openfhe header" \
  "translate" \
  "--emit-openfhe-pke-header" \
  "04_mlir_openfhe.mlir" \
  "04_emit_openfhe.h" \

cd $cwd
