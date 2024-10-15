import json
import subprocess
import os
import pdb

def print_step_name(step_name):
    try:
        subprocess.run(['figlet', step_name], check=True)
    except FileNotFoundError:
        line = '-' * len(step_name)
        print(f"\n{line}\n{step_name}\n{line}\n")

def run_command(command, args, working_dir, input_mlir):
    formatted_args = [arg.format(working_dir=working_dir, input_mlir=input_mlir) for arg in args]
    full_command = f"{command} {' '.join(formatted_args)}"
    subprocess.run(full_command, shell=True, check=True)

def get_absolute_path(path):
    return path if os.path.isabs(path) else os.path.abspath(path)

def main():
    # Load the JSON file
    with open('passes.json', 'r') as f:
        passes = json.load(f)['passes']

    # Set directories
    heir_dir = '/home/ubuntu/heir'
    working_dir = '/home/ubuntu/simplemlir'
    input_mlir = 'handwritten_matmul_flat.mlir'

    # Change to heir directory
    os.chdir(heir_dir)

    # Run each pass
    for pass_info in passes:
        print_step_name(pass_info['name'])
        run_command(pass_info['command'], pass_info['args'], working_dir, input_mlir)

if __name__ == "__main__":
    main()
