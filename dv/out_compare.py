#!/usr/bin/env python3
# Written by Claude.ai

import argparse
import sys

def compare_simulation_with_model(simulation_output_path, expected_output_path):
    """
    Compare hardware simulation output against expected output from a software model.

    Purpose:
    This function validates that the byte-by-byte output from hardware simulation
    matches the expected output pattern defined by a software model. The software
    model can specify either exact byte values (as 2-digit hex) or wildcards ('X')
    that match any byte from the simulation.

    Parameters:
    - simulation_output_path: Path to file containing captured hardware simulation output
                             (File A, each line contains a single 2-digit hex value)
    - expected_output_path: Path to file containing expected output from software model
                           (File B, each line contains either a 2-digit hex value or 'X')

    Rules:
    - Simulation output (File A) contains only 2-digit hex values (00-FF)
    - Expected output (File B) contains either 2-digit hex values or 'X' wildcards
    - Where expected output has a hex value, simulation must match exactly
    - Where expected output has 'X', any simulation value is accepted
    - The simulation may produce more output than expected (trailing bytes ignored)
    - The expected output must not be longer than simulation output (incomplete simulation)

    Returns:
    - True if simulation output matches the expected pattern
    - False if there's a mismatch or validation error
    """
    try:
        with open(simulation_output_path, 'r') as sim_file, open(expected_output_path, 'r') as model_file:
            sim_lines = sim_file.readlines()
            model_lines = model_file.readlines()

            # Check if expected output is longer than simulation output (error condition)
            if len(model_lines) > len(sim_lines):
                print(f"Error: Expected output has more bytes ({len(model_lines)}) than simulation output ({len(sim_lines)})")
                print("This suggests the hardware simulation terminated prematurely or is incomplete")
                return False

            # Compare each byte in expected output with corresponding byte in simulation
            for i, model_line in enumerate(model_lines):
                model_byte = model_line.strip()
                sim_byte = sim_lines[i].strip()

                # Validate format of simulation output (should be 2-digit hex)
                if not (len(sim_byte) == 2 and all(c in '0123456789abcdefABCDEF' for c in sim_byte)):
                    print(f"Error at byte {i+1} in simulation output: '{sim_byte}' is not a valid 2-digit hex value")
                    print("Simulation output should only contain 2-digit hex values (00-FF)")
                    return False

                # Check if expected byte is a wildcard or a matching hex value
                if model_byte == 'X':
                    # Wildcard - simulation can have any value here
                    continue
                elif len(model_byte) == 2 and all(c in '0123456789abcdefABCDEF' for c in model_byte):
                    # Expected specific hex value - must match simulation
                    if model_byte.lower() != sim_byte.lower():
                        print(f"Mismatch at byte {i+1}: Simulation produced '{sim_byte}', model expected '{model_byte}'")
                        return False
                else:
                    print(f"Error at byte {i+1} in expected output: '{model_byte}' is neither a valid 2-digit hex value nor 'X'")
                    print("Expected output should only contain 2-digit hex values (00-FF) or 'X' wildcards")
                    return False

            if len(sim_lines) > len(model_lines):
                extra_bytes = len(sim_lines) - len(model_lines)
                print(f"Note: Simulation produced {extra_bytes} additional byte(s) beyond what was specified in the model")
                print("These additional bytes were ignored in the comparison")

            print("Verification successful: Hardware simulation output matches "
                  "the expected pattern from software model "
                  f"{len(model_lines)} bytes matched")
            return True

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that both simulation output and expected output files exist")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Verify hardware simulation output against expected software model output'
    )

    parser.add_argument('-s', '--simulation',
                        dest='simulation_file',
                        required=True,
                        help='File containing hardware simulation output (one hex byte per line)')

    parser.add_argument('-e', '--expected',
                        dest='expected_file',
                        required=True,
                        help='File containing expected output from software model (hex bytes or X wildcards)')

    parser.add_argument('-q', '--quiet',
                        action='store_true',
                        help='Suppress informational messages, show only errors')

    args = parser.parse_args()

    # If quiet mode is enabled, redirect stdout temporarily
    if args.quiet:
        original_stdout = sys.stdout
        sys.stdout = open('/dev/null', 'w')

    try:
        result = compare_simulation_with_model(args.simulation_file, args.expected_file)

        # Restore stdout if it was redirected
        if args.quiet:
            sys.stdout.close()
            sys.stdout = original_stdout

        # Always show the final result, even in quiet mode
        if args.quiet:
            if result:
                print("Verification successful")
            else:
                print("Verification failed")

        sys.exit(0 if result else 1)
    finally:
        # Ensure stdout is restored if an exception occurs
        if args.quiet and sys.stdout != sys.__stdout__:
            sys.stdout.close()
            sys.stdout = original_stdout

if __name__ == "__main__":
    main()
