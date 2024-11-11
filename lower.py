import json
import subprocess
import os
import pdb
import argparse

def print_step_name(step_name):
    try:
        subprocess.run(['figlet', '-w 200', step_name], check=True)
    except FileNotFoundError:
        line = '-' * len(step_name)
        print(f"\n{line}\n{step_name}\n{line}\n")

def run_command(command, args, working_dir, input_mlir, index, step_name, extension='mlir'):
    # Remove spaces and invalid characters from step_name
    step_name = step_name.replace(" ", "_").replace("-", "_")
    
    # Format arguments and add output redirection
    output_file = os.path.join(working_dir, f"{index:02d}_{step_name}.{extension}")
    error_file = os.path.join(working_dir, f"{index:02d}_{step_name}.err")
    
    # Correct the command to have a single output redirection
    full_command = f"{command} {' '.join(args)} {input_mlir} > {output_file} 2> {error_file}"
    
    # Execute the command
    subprocess.run(full_command, shell=True, check=True)
    
    return output_file  # Return the output file path for the next pass

def get_absolute_path(path):
    return path if os.path.isabs(path) else os.path.abspath(path)

def main():
    # argparse stuff
    parser = argparse.ArgumentParser(description='Lower a program written in MLIR dialects using the passes specified in a json file')
    parser.add_argument('input_mlir', type=str, help='The input MLIR file to lower', default='input.mlir')
    parser.add_argument('--passes', type=str, help='The passes to use', default='passes.json')
    parser.add_argument('--clean', type=bool, help='Clean the working directory. This deletes the output files of the previous run, which will start with 01, 02, .... Be careful.', default=False, const=True, nargs='?')
    args = parser.parse_args()

    # clean
    if args.clean:
        subprocess.run('rm -f 0*err 0*mlir', shell=True, check=True)

    # Load the JSON file
    with open(args.passes, 'r') as f:
        passes = json.load(f)['passes']

    # Set directories
    heir_dir = get_absolute_path('/home/ubuntu/heir')
    working_dir = get_absolute_path('/home/ubuntu/simplemlir')
    input_mlir = get_absolute_path(args.input_mlir)

    # Change to heir directory
    os.chdir(heir_dir)

    # Run each pass
    for index, pass_info in enumerate(passes):

        # Print the step name with index prefix
        print_step_name(f"( {index + 1} ) {pass_info['name']}")

        if 'outputs' in pass_info:
            # Handle multiple outputs case
            for output in pass_info['outputs']:
                run_command(
                    pass_info['command'],
                    output['args'],
                    working_dir,
                    input_mlir,
                    index+1,
                    f"{pass_info['name']}_{output['extension']}",
                    extension=output['extension']
                )
            # Don't continue the pipeline after generating final C++ outputs
            break
        else:
            # Regular MLIR pass
            output_mlir = run_command(
                pass_info['command'],
                pass_info['args'],
                working_dir,
                input_mlir,
                index+1,
                pass_info['name']
            )

            # Check if the output file differs from the input file
            if input_mlir.endswith('.mlir'):
                with open(input_mlir, 'r') as f:
                    input_content = f.read()
                with open(output_mlir, 'r') as f:
                    output_content = f.read()
                if input_content == output_content:
                    print(f"Output file {output_mlir} is identical to input file {input_mlir}")
                    break

            # Update input_mlir for the next pass
            input_mlir = output_mlir


if __name__ == "__main__":
    main()
