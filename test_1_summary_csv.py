"""
Summarize num_trials and total_time from CSV files in a directory.

For each CSV file, calculates the average num_trials and average total_time
across all branches and creates a summary CSV.

Usage:
    python summary_num_trials_and_tot_time.py benchmark_log_1_biased_test
    python summary_num_trials_and_tot_time.py benchmark_log_2_biased_test
    python summary_num_trials_and_tot_time.py benchmark_log_ga_biased_test
"""

import csv
import os
import sys
from pathlib import Path


def summarize_directory(dir_name: str):
    """
    Summarize all CSV files in a directory.
    
    Args:
        dir_name: Directory containing CSV files to summarize
    
    Creates:
        summary_result/summary_report_<dir_name>.csv
    """
    dir_path = Path(dir_name)
    
    if not dir_path.exists():
        print(f" Error: Directory '{dir_name}' does not exist!")
        sys.exit(1)
    
    # Find all CSV files in the directory
    csv_files = [f for f in dir_path.glob("*.csv") if f.name != "test_config.json"]
    
    if not csv_files:
        print(f" Error: No CSV files found in '{dir_name}'")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f" Summarizing CSV files in: {dir_name}")
    print(f"{'='*80}")
    print(f"Found {len(csv_files)} CSV files\n")
    
    summary_rows = []
    
    for csv_file in sorted(csv_files):
        file_name = csv_file.stem  # Get filename without extension
        
        num_trials_list = []
        total_time_list = []
        
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                
                for row in reader:
                    # Parse num_trials (could be int or float)
                    try:
                        num_trials = float(row['num_trials'])
                        num_trials_list.append(num_trials)
                    except (KeyError, ValueError) as e:
                        print(f"  Warning: Could not parse num_trials in {csv_file.name}, row: {row}")
                        continue
                    
                    # Parse total_time (could be string like "20.000")
                    try:
                        total_time = float(row['total_time'])
                        total_time_list.append(total_time)
                    except (KeyError, ValueError) as e:
                        print(f"  Warning: Could not parse total_time in {csv_file.name}, row: {row}")
                        continue
            
            # Calculate averages
            if num_trials_list and total_time_list:
                avg_num_trials = sum(num_trials_list) / len(num_trials_list)
                avg_total_time = sum(total_time_list) / len(total_time_list)
                
                summary_rows.append({
                    'file_name': file_name,
                    'avg_num_trials': round(avg_num_trials, 2),
                    'avg_total_time': round(avg_total_time, 3),
                    'num_branches': len(num_trials_list)
                })
                
                print(f"[OK] {file_name:<30} branches={len(num_trials_list):<3} avg_trials={avg_num_trials:>8.2f} avg_time={avg_total_time:>8.3f}s")
            else:
                print(f"  {file_name:<30} (no data)")
                
        except Exception as e:
            print(f" Error processing {csv_file.name}: {e}")
            continue
    
    if not summary_rows:
        print(f"\n No valid data found to summarize!")
        sys.exit(1)
    
    # Create summary_result directory
    summary_dir = Path("summary_result")
    summary_dir.mkdir(exist_ok=True)
    
    # Create output CSV
    output_csv = summary_dir / f"summary_report_{dir_path.name}.csv"
    
    with open(output_csv, 'w', newline='') as f:
        fieldnames = ['file_name', 'avg_num_trials', 'avg_total_time', 'num_branches']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(summary_rows)
    
    print(f"\n{'='*80}")
    print(f" Summary written to: {output_csv}")
    print(f"{'='*80}")
    print(f"Total files summarized: {len(summary_rows)}")
    
    # Print summary statistics
    total_avg_trials = sum(r['avg_num_trials'] for r in summary_rows)
    total_avg_time = sum(r['avg_total_time'] for r in summary_rows)
    
    print(f"\n OVERALL STATISTICS")
    print(f"{'-'*80}")
    print(f"Total average num_trials across all files: {total_avg_trials:.2f}")
    print(f"Total average total_time across all files: {total_avg_time:.3f}s")
    print(f"Grand average num_trials per file: {total_avg_trials/len(summary_rows):.2f}")
    print(f"Grand average total_time per file: {total_avg_time/len(summary_rows):.3f}s")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python summary_num_trials_and_tot_time.py <directory>")
        print("\nExamples:")
        print("  python summary_num_trials_and_tot_time.py benchmark_log_1_biased_test")
        print("  python summary_num_trials_and_tot_time.py benchmark_log_2_biased_test")
        print("  python summary_num_trials_and_tot_time.py benchmark_log_ga_biased_test")
        sys.exit(1)
    
    directory = sys.argv[1]
    summarize_directory(directory)

