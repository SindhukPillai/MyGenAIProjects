import pandas as pd
import os

def preprocess_and_analyze(input_path, output_path, summary_path):
    # Step 1: Load the Excel file safely
    df = pd.read_excel(input_path, sheet_name=0, engine="openpyxl")

    # Step 2: Clean and rename columns
    original_columns = df.columns.tolist()
    renamed_columns = [
        col.strip()
        .replace("\n", " ")
        .replace("\\", "")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace(":", "")
        .replace("-", "_")
        .replace("  ", " ")
        .replace(" ", "_")
        for col in original_columns
    ]
    df.columns = renamed_columns

    # Step 3: Normalize text in object columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.replace("\n", " ").str.replace("\r", " ")

    # Step 4: Group and summarize values
    summary_data = []
    for col in df.columns:
        value_counts = df[col].value_counts(dropna=False).head(5)
        summary_data.append({
            "Column": col,
            "Unique Values": df[col].nunique(dropna=False),
            "Top 1": f"{repr(value_counts.index[0])}: {value_counts.iloc[0]}" if len(value_counts) > 0 else "",
            "Top 2": f"{repr(value_counts.index[1])}: {value_counts.iloc[1]}" if len(value_counts) > 1 else "",
            "Top 3": f"{repr(value_counts.index[2])}: {value_counts.iloc[2]}" if len(value_counts) > 2 else "",
            "Top 4": f"{repr(value_counts.index[3])}: {value_counts.iloc[3]}" if len(value_counts) > 3 else "",
            "Top 5": f"{repr(value_counts.index[4])}: {value_counts.iloc[4]}" if len(value_counts) > 4 else "",
        })

    summary_df = pd.DataFrame(summary_data)

    # Step 5: Save cleaned file and summary
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    summary_df.to_excel(summary_path, index=False)

    print("✅ Preprocessing Complete")
    print(f"Cleaned file saved to: {output_path}")
    print(f"Column summary saved to: {summary_path}")

# Example usage
input_file = r"Raw\GEHC_IAF_Tracker_History_Raw.xlsx"
output_file = r"Training\GEHC_IAF_Tracker_History_Cleaned.xlsx"
summary_file = r"Training\GEHC_IAF_Column_Summary.xlsx"

preprocess_and_analyze(input_file, output_file, summary_file)
