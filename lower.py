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

def run_command(command, args, working_dir, input_mlir, index, step_name):
    # Remove spaces and invalid characters from step_name
    step_name = step_name.replace(" ", "_").replace("-", "_")
    
    # Format arguments and add output redirection
    output_file = os.path.join(working_dir, f"{index:02d}_{step_name}.mlir")
    error_file = os.path.join(working_dir, f"{index:02d}_{step_name}.err")
    
    # Correct the command to have a single output redirection
    full_command = f"{command} {' '.join(args)} {input_mlir} > {output_file} 2> {error_file}"
    
    # Execute the command
    # breakpoint()
    subprocess.run(full_command, shell=True, check=True)
    
    return output_file  # Return the output file path for the next pass

def get_absolute_path(path):
    return path if os.path.isabs(path) else os.path.abspath(path)

def main():
    # Load the JSON file
    with open('passes.json', 'r') as f:
        passes = json.load(f)['passes']

    # Set directories
    heir_dir = get_absolute_path('/home/ubuntu/heir')
    working_dir = get_absolute_path('/home/ubuntu/simplemlir')
    input_mlir = get_absolute_path('handwritten_matmul_flat.mlir')

    # Change to heir directory
    os.chdir(heir_dir)

    # Run each pass
    for index, pass_info in enumerate(passes):
        print_step_name(pass_info['name'])
        input_mlir = run_command(pass_info['command'], pass_info['args'], working_dir, input_mlir, index+1, pass_info['name'])

if __name__ == "__main__":
    main()
