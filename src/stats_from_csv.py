import sys
import os
import pandas as pd
import numpy as np

#/Users/sylviadong/Documents/Resume_Bias/Bias_code/Ent_plots_compare/ [filename]

def calculate_statistics(file_path):
    scores_path = os.path.join(os.path.dirname(__file__), 'scores.txt')
    try:
        # Load CSV file
        df = pd.read_csv(file_path)

        # Keep only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            # append message to scores.txt
            with open(scores_path, 'a', encoding='utf-8') as out:
                out.write(f"\nNo numeric columns found in the CSV file: {file_path}\n")
            return

        lines = []
        lines.append(f"\nStatistical Summary for: {file_path}\n")

        for column in numeric_df.columns:
            data = numeric_df[column].dropna()

            mean = data.mean()
            median = data.median()
            std_dev = data.std()
            q1 = data.quantile(0.25)
            q2 = data.quantile(0.50)
            q3 = data.quantile(0.75)
            skewness = data.skew()

            lines.append(f"Column: {column}")
            lines.append(f"  Mean: {mean}")
            lines.append(f"  Median: {median}")
            lines.append(f"  Standard Deviation: {std_dev}")
            lines.append(f"  Q1 (25%): {q1}")
            lines.append(f"  Q2 (50%): {q2}")
            lines.append(f"  Q3 (75%): {q3}")
            lines.append(f"  Skewness: {skewness}")
            lines.append("-" * 40)

        # append all lines to scores.txt
        with open(scores_path, 'a', encoding='utf-8') as out:
            out.write("\n".join(lines) + "\n")

    except FileNotFoundError:
        print("Error: File not found.")
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python /Users/sylviadong/Documents/Resume_Bias/Bias_code/stats_from_csv.py <path_to_csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    calculate_statistics(file_path)
