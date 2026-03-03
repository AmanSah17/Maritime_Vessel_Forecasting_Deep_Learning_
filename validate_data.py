import pandas as pd
import sys

def main():
    try:
        path = r"F:\PyTorch_GPU\vessel_trajectory_prediction\Data\region_1_q1_merged_renamed.parquet"
        df = pd.read_parquet(path)
        
        # Cleanly print columns
        columns = df.columns.tolist()
        print(f"Total Rows: {len(df)}")
        print(f"Columns: {columns}")
        
        # Identify Time and MMSI columns case-insensitively
        time_col = next((c for c in columns if c.lower() in ['timestamp', 'basedatetime', 'time']), None)
        id_col = next((c for c in columns if c.lower() in ['mmsi', 'id', 'vessel_id']), None)
        
        if time_col and id_col:
            print(f"Using Time Column: '{time_col}' and ID Column: '{id_col}'")
            
            # Ensure datetime type
            df[time_col] = pd.to_datetime(df[time_col])
            df = df.sort_values(by=[id_col, time_col])
            
            # Calculate time difference in minutes
            diffs = df.groupby(id_col)[time_col].diff().dt.total_seconds() / 60.0
            
            print("\n--- Time Interval Analysis (Minutes) ---")
            print(diffs.describe())
            print("\nTop 5 Most Common Intervals (Minutes):")
            print(diffs.value_counts().head(5))
        else:
            print("Could not find standard time or MMSI columns!")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
