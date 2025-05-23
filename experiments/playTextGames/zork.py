import subprocess
import sys
import time
import os

def run_zork1():
    zork_path = "/home/arturvrsampaio/zork/zork1/DATA/ZORK1.DAT"

    # Verify the Zork 1 data file exists
    if not os.path.isfile(zork_path):
        print(f"Error: Zork 1 data file not found at {zork_path}")
        sys.exit(1)

    # Verify Frotz is installed
    try:
        subprocess.run(["frotz", "--version"], capture_output=True, text=True)
    except FileNotFoundError:
        print("Error: Frotz not found. Please install Frotz (e.g., 'sudo apt-get install frotz' on Linux).")
        sys.exit(1)

    # Start Frotz with the Zork 1 data file
    process = subprocess.Popen(
        ["frotz", zork_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    print("Starting Zork 1...")

    # Read initial output to confirm the game started
    initial_output = ""
    stderr_output = ""
    start_time = time.time()
    while time.time() - start_time < 5:  # Wait up to 5 seconds for initial output
        line = process.stdout.readline()
        if line:
            initial_output += line
            print(line, end='', flush=True)
        # Check stderr for errors
        err_line = process.stderr.readline()
        if err_line:
            stderr_output += err_line
        if "ZORK I: The Great Underground Empire" in initial_output:
            break
        time.sleep(0.1)

    if not initial_output:
        print("Error: No output from Zork 1. Frotz may have failed to start.")
        if stderr_output:
            print("Frotz error output:\n" + stderr_output)
        process.terminate()
        sys.exit(1)

    # Main interaction loop
    while True:
        # Read user input
        user_input = input()
        if user_input.lower() in ['quit', 'exit']:
            break

        # Send input to Frotz
        process.stdin.write(user_input + '\n')
        process.stdin.flush()

        # Read and display output until the next prompt
        while True:
            line = process.stdout.readline()
            if not line:  # Process ended
                print("Error: Zork 1 process terminated unexpectedly.")
                # Check stderr for any final errors
                stderr_output = process.stderr.read()
                if stderr_output:
                    print("Frotz error output:\n" + stderr_output)
                process.terminate()
                sys.exit(1)
            print(line, end='', flush=True)
            if line.strip() == ">":  # Zork prompt
                break
            time.sleep(0.01)

    # Clean up
    process.terminate()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
    print("Zork 1 terminated.")

if __name__ == "__main__":
    run_zork1()