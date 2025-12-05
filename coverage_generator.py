import csv
import os
import subprocess
import sys

def validate_test_case(module_name: str, func_name: str, args_str: str) -> bool:
    """
    Test if a test case causes runtime errors (ZeroDivisionError, etc).
    Returns True if valid (no errors), False if invalid (causes exceptions).
    """
    try:
        import sys
        import importlib
        
        # Add benchmark directory to path temporarily
        benchmark_dir = os.path.abspath('benchmark')
        if benchmark_dir not in sys.path:
            sys.path.insert(0, benchmark_dir)
        
        # Import and reload to get fresh module
        if module_name in sys.modules:
            module = importlib.reload(sys.modules[module_name])
        else:
            module = __import__(module_name)
        
        func = getattr(module, func_name)
        
        # Parse and execute
        args = eval(f"({args_str},)" if args_str else "()")
        func(*args)
        return True  # No exception = valid test
        
    except (ZeroDivisionError, ValueError, OverflowError, IndexError) as e:
        # These are INVALID inputs - filter them out
        print(f"  ⚠️  Invalid test: {func_name}({args_str}) → {type(e).__name__}")
        return False
        
    except Exception as e:
        # Other exceptions might be expected behavior
        # Keep the test case
        return True


def main(dir_name: str):
    csv_files = sorted([f for f in os.listdir(dir_name) if f.endswith(".csv")])  # ✅ Sort alphabetically
    os.makedirs("benchmark", exist_ok=True)
    os.makedirs("coverage_result", exist_ok=True)

    output_rows = []
    total_invalid = 0

    for csv_file in csv_files:
        program_name = csv_file[:-4]   # remove .csv
        csv_path = os.path.join(dir_name, csv_file)

        tests = []
        invalid_count = 0
        
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                func = row["function"]
                sol = row["best_solution"][1:-1]
                
                # ✅ Validate test case before adding
                if validate_test_case(program_name, func, sol):
                    tests.append((func, idx, sol))
                else:
                    invalid_count += 1
                    total_invalid += 1
        
        if invalid_count > 0:
            print(f"  🚫 Filtered {invalid_count} invalid test(s) from {program_name}")

        # Create temporary test script
        test_filename = f"./benchmark/test_{program_name}.py"
        with open(test_filename, "w", encoding="utf-8") as t:
            t.write(f"import {program_name}\n\n")
            for func, idx, sol in tests:
                t.write(f"def test_{func}_{idx}():\n")
                t.write(f"    {program_name}.{func}({sol})\n\n")

        # Run coverage
        try:
            subprocess.run(
                ["coverage", "run", "--branch", "-m", "pytest", test_filename],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            result = subprocess.run(
                ["coverage", "report", "-m"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        except subprocess.CalledProcessError as e:
            print("Coverage execution failed:", e.stderr)
            continue

        # Parse coverage output
        coverage_lines = result.stdout.splitlines()
        for line in coverage_lines:
            if program_name in line:
                parts = line.split()
                miss = parts[2]
                branch = parts[3]
                coverage_percent = parts[5]
                missing_line = " ".join(parts[6:]) if len(parts) > 6 else None
                output_rows.append((program_name, coverage_percent, miss, branch, missing_line))
                break

        # Clean temp script
        print(test_filename)
        # os.remove(test_filename)

        # print(f"Complete coverage checking for {program_name}: miss / branch = {miss}{f"({missing_line})" if missing_line else ""} / {branch}")
        missing_part = f"({missing_line})" if missing_line else ""

        print(f"Complete coverage checking for {program_name}: miss / branch = {miss}{missing_part} / {branch}")

    # Sort output rows alphabetically by file name
    output_rows.sort(key=lambda x: x[0])  # Sort by first element (file_name)
    
    # Write final CSV
    out_path = f"coverage_result/coverage_report_{dir_name}.csv"
    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["file_name", "total_coverage", "misses", "branches", "missing_line"])
        writer.writerows(output_rows)

    print(f"Coverage report written to {out_path}")

    total = 0
    for (_, percentage, _, _, _) in output_rows:
        total += int(percentage[:-1])
    
    print(f"\n{'='*80}")
    print(f"📊 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Total invalid tests filtered: {total_invalid}")
    print(f"Final coverage: {total}% / {len(output_rows) * 100}%")
    print(f"Average coverage per file: {total / len(output_rows):.1f}%")
    print(f"{'='*80}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python coverage_generator.py <directory_with_csvs>")
        sys.exit(1)

    main(sys.argv[1])
