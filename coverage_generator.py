import csv
import os
import subprocess
import sys

def main(dir_name: str):
    csv_files = [f for f in os.listdir(dir_name) if f.endswith(".csv")]
    os.makedirs("benchmark", exist_ok=True)
    os.makedirs("coverage_result", exist_ok=True)

    output_rows = []

    for csv_file in csv_files:
        program_name = csv_file[:-4]   # remove .csv
        csv_path = os.path.join(dir_name, csv_file)

        tests = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                func = row["function"]
                sol = row["best_solution"][1:-1]
                tests.append((func, idx, sol))

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
                ["coverage", "report"],
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
                coverage_percent = parts[-1]
                output_rows.append((program_name, coverage_percent))
                break

        # Clean temp script
        os.remove(test_filename)

    # Write final CSV
    out_path = f"coverage_result/coverage_report_{dir_name}.csv"
    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["file_name", "total_coverage"])
        writer.writerows(output_rows)

    print(f"Coverage report written to {out_path}")

    total = 0
    for (_, percentage) in output_rows:
        total += int(percentage[:-1])
    
    print(f"Final coverage: {total}% / {len(output_rows) * 100}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python coverage_generator.py <directory_with_csvs>")
        sys.exit(1)

    main(sys.argv[1])
