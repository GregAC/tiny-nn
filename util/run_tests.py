#!/usr/bin/env python3
# Written by Claude.ai

import argparse
import os
import subprocess
import tempfile
import sys
from typing import List, Tuple, Dict, Literal
from abc import ABC, abstractmethod
import re

# Define a type for test status
TestStatus = Literal["PASS", "FAIL", "SIM_ERROR"]

# ANSI color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

# Regular expression to strip ANSI color codes
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


class SimulatorRunner(ABC):
    """Abstract base class for simulator runners."""

    @abstractmethod
    def run(self, input_file: str, output_file: str, verbose: bool) -> bool:
        """
        Run the simulation with the given input file.

        Args:
            input_file: Path to the input test vector file
            output_file: Path where the output should be written
            verbose: Whether to print verbose output

        Returns:
            True if simulation succeeded, False otherwise
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this simulator."""
        pass

    @abstractmethod
    def check_binary(self) -> Tuple[bool, str]:
        """
        Check if the simulator binary exists and is executable.

        Returns:
            Tuple of (success, error_message)
        """
        pass


class RTLSimulatorRunner(SimulatorRunner):
    """Runner for RTL (Verilator) simulation."""

    def __init__(self, binary_path: str):
        self.binary_path = binary_path

    def run(self, input_file: str, output_file: str, verbose: bool) -> bool:
        cmd = [
            self.binary_path,
            f"+test_data={input_file}",
            f"+out={output_file}"
        ]

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error: RTL simulation failed with exit code {e.returncode}")
            return False

    def get_name(self) -> str:
        return "RTL (Verilator)"

    def check_binary(self) -> Tuple[bool, str]:
        if not os.path.exists(self.binary_path):
            return (False, f"RTL simulator not found at '{self.binary_path}'\n"
                          f"Build it with: make -C dv sim")
        if not os.access(self.binary_path, os.X_OK):
            return (False, f"RTL simulator at '{self.binary_path}' is not executable")
        return (True, "")


class ModelSimulatorRunner(SimulatorRunner):
    """Runner for the Rust model in simulate mode."""

    def __init__(self, binary_path: str = None, use_cargo: bool = False, model_dir: str = "model"):
        self.binary_path = binary_path
        self.use_cargo = use_cargo
        self.model_dir = model_dir

    def run(self, input_file: str, output_file: str, verbose: bool) -> bool:
        if self.use_cargo:
            cmd = [
                "cargo", "run", "--release", "--manifest-path",
                os.path.join(self.model_dir, "Cargo.toml"),
                "--", "simulate", "--input", input_file, "--output", output_file
            ]
        else:
            cmd = [
                self.binary_path,
                "simulate", "--input", input_file, "--output", output_file
            ]

        if verbose:
            print(f"Running: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL if not verbose else None,
                stderr=subprocess.DEVNULL if not verbose else None
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error: Model simulation failed with exit code {e.returncode}")
            return False

    def get_name(self) -> str:
        if self.use_cargo:
            return "Model (cargo)"
        return "Model"

    def check_binary(self) -> Tuple[bool, str]:
        if self.use_cargo:
            cargo_toml = os.path.join(self.model_dir, "Cargo.toml")
            if not os.path.exists(cargo_toml):
                return (False, f"Cargo.toml not found at '{cargo_toml}'\n"
                              f"Specify --model-dir to point to the model directory")
            return (True, "")
        else:
            if not os.path.exists(self.binary_path):
                return (False, f"Model binary not found at '{self.binary_path}'\n"
                              f"Build it with: cargo build --release --manifest-path model/Cargo.toml\n"
                              f"Or use --cargo to run via cargo")
            if not os.access(self.binary_path, os.X_OK):
                return (False, f"Model binary at '{self.binary_path}' is not executable")
            return (True, "")


def create_runner(mode: str, simulator_path: str, use_cargo: bool, model_dir: str) -> SimulatorRunner:
    """Factory function to create the appropriate simulator runner."""
    if mode == "rtl":
        return RTLSimulatorRunner(simulator_path)
    elif mode == "model":
        return ModelSimulatorRunner(simulator_path, use_cargo, model_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run simulations and compare outputs against expected results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # RTL tests (default)
  %(prog)s --mode model             # Model tests with pre-built binary
  %(prog)s --mode model --cargo     # Model tests using cargo
  %(prog)s -m model -t dv/test_vectors -v  # Model tests, verbose
"""
    )
    parser.add_argument('-m', '--mode', choices=['rtl', 'model'], default='rtl',
                        help="Simulation mode: 'rtl' (default) or 'model'")
    parser.add_argument('-t', '--test_dir', default=None,
                        help='Directory containing test vectors (default: dv/test_vectors)')
    parser.add_argument('-s', '--simulator', default=None,
                        help='Path to simulator binary (auto-selected by mode if not specified)')
    parser.add_argument('--cargo', action='store_true',
                        help='Model mode: use "cargo run" instead of pre-built binary')
    parser.add_argument('--model-dir', default='model',
                        help='Model mode: directory containing Cargo.toml (default: model)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--no-color', action='store_true',
                        help='Disable colored output')
    return parser.parse_args()


def get_default_paths(mode: str) -> Tuple[str, str]:
    """Get default test_dir and simulator path for a given mode."""
    test_dir = "dv/test_vectors"

    if mode == "rtl":
        simulator = "./obj_dir/Vtiny_nn_top_tb"
    else:  # model
        simulator = "./model/target/release/model"

    return (test_dir, simulator)


def find_test_files(test_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find all test input files and their corresponding expected output files.

    Returns a list of tuples (test_name, input_file, expected_output_file)
    """
    test_files = []

    # Ensure the test directory exists
    if not os.path.isdir(test_dir):
        print(f"Error: Test directory '{test_dir}' not found.")
        sys.exit(1)

    # Find all input files matching the pattern test_*.hex
    for file in os.listdir(test_dir):
        if file.startswith('test_') and file.endswith('.hex') and not file.endswith('_expected.hex'):
            input_file = os.path.join(test_dir, file)

            # Extract test name - remove 'test_' prefix and '.hex' suffix
            test_name = file[5:-4]  # Extract name between 'test_' and '.hex'

            # Construct expected output filename
            expected_file = os.path.join(test_dir, f"test_{test_name}_expected.hex")

            # Check if expected output file exists
            if os.path.isfile(expected_file):
                test_files.append((test_name, input_file, expected_file))
            else:
                print(f"Warning: Expected output file '{expected_file}' not found for input '{input_file}'.")

    return test_files


def run_simulation(runner: SimulatorRunner, input_file: str, verbose: bool) -> Tuple[bool, str]:
    """
    Run the simulation with the given input file.

    Returns a tuple of (success, output_file_path).
    If simulation fails, success will be False and output_file_path will be None.
    """
    # Create a temporary file for the output
    with tempfile.NamedTemporaryFile(delete=False, suffix='.hex') as temp_file:
        output_file = temp_file.name

    # Run the simulation
    success = runner.run(input_file, output_file, verbose)

    if success:
        return (True, output_file)
    else:
        if os.path.exists(output_file):
            os.unlink(output_file)
        return (False, None)


def compare_outputs(sim_output: str, expected_output: str, verbose: bool) -> bool:
    """
    Compare the simulation output with the expected output.

    Returns True if the outputs match, False otherwise.
    """
    # Find out_compare.py relative to this script's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    compare_script = os.path.join(script_dir, 'out_compare.py')

    cmd = [
        compare_script,
        '-s', sim_output,
        '-e', expected_output
    ]

    if verbose:
        print(f"Running: {' '.join(cmd)}")

    # Run the comparison
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from a string."""
    return ANSI_ESCAPE.sub('', text)


def colorize(text: str, color: str, use_color: bool = True) -> str:
    """Apply color to text if use_color is True."""
    if use_color:
        return f"{color}{text}{Colors.RESET}"
    return text


def print_results_table(results: Dict[str, TestStatus], runner_name: str, use_color: bool = True):
    """Print a formatted ASCII table of the test results."""
    # Print which simulator was used
    print(f"\nSimulator: {runner_name}")

    # Find the longest test name for formatting
    max_name_len = max(len(name) for name in results.keys())
    result_col_width = 10  # Width for the result column

    # Create the table structure
    separator = f"+{'-' * (max_name_len + 2)}+{'-' * (result_col_width + 2)}+"

    # Print the table header
    header_test = "Test Name"
    header_result = "Result"

    if use_color:
        header_test = colorize(header_test, Colors.BOLD)
        header_result = colorize(header_result, Colors.BOLD)

    print(separator)
    print(f"| {header_test:<{max_name_len+8}} | {header_result:<{result_col_width+8}} |")
    print(separator)

    # Print each result
    pass_count = 0
    sim_error_count = 0

    for name, status in results.items():
        result_text = status
        if status == "PASS":
            pass_count += 1
            result_styled = colorize(result_text, Colors.GREEN, use_color)
        elif status == "SIM_ERROR":
            sim_error_count += 1
            result_styled = colorize(result_text, Colors.YELLOW, use_color)
        else:  # FAIL
            result_styled = colorize(result_text, Colors.RED, use_color)

        print(f"| {name:<{max_name_len}} | {result_styled}{' ' * (result_col_width - len(result_text))} |")

    # Print the footer
    print(separator)

    # Print summary
    total = len(results)
    fail_count = total - pass_count - sim_error_count

    pass_percentage = pass_count/total*100 if total > 0 else 0

    summary_text = f"Summary: {pass_count}/{total} tests passed ({pass_percentage:.1f}%)"
    print(colorize(summary_text, Colors.GREEN if pass_count == total else Colors.RESET, use_color))

    if sim_error_count > 0:
        sim_error_text = f"         {sim_error_count}/{total} simulation errors"
        print(colorize(sim_error_text, Colors.YELLOW, use_color))

    if fail_count > 0:
        fail_text = f"         {fail_count}/{total} comparison failures"
        print(colorize(fail_text, Colors.RED, use_color))


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()

    # Get default paths for the selected mode
    default_test_dir, default_simulator = get_default_paths(args.mode)

    # Use provided paths or defaults
    test_dir = args.test_dir if args.test_dir else default_test_dir
    simulator_path = args.simulator if args.simulator else default_simulator

    # Determine if we should use colors
    use_color = not args.no_color and sys.stdout.isatty()

    # Create the appropriate runner
    runner = create_runner(args.mode, simulator_path, args.cargo, args.model_dir)

    # Check if the simulator binary exists
    binary_ok, error_msg = runner.check_binary()
    if not binary_ok:
        print(f"Error: {error_msg}")
        sys.exit(1)

    if args.verbose:
        print(f"Using simulator: {runner.get_name()}")

    # Find test files
    test_files = find_test_files(test_dir)

    if not test_files:
        print("No test files found.")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(test_files)} test files.")

    # Run simulations and compare outputs
    results = {}

    for test_name, input_file, expected_file in test_files:
        if args.verbose:
            print(f"\nProcessing test: {test_name}")

        # Run simulation
        sim_success, output_file = run_simulation(runner, input_file, args.verbose)

        if not sim_success:
            results[test_name] = "SIM_ERROR"
            continue

        # Compare outputs
        comparison_passed = compare_outputs(output_file, expected_file, args.verbose)
        results[test_name] = "PASS" if comparison_passed else "FAIL"

        # Clean up temporary output file
        os.unlink(output_file)

    # Print results
    print_results_table(results, runner.get_name(), use_color)

if __name__ == "__main__":
    main()
