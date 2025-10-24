import pandas as pd
from pathlib import Path

folder = Path("results/pubmed_eval")

csv_files = sorted(folder.glob("*.csv"))

# list to store dataframes
dfs = []

for f in csv_files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = f.name  
        dfs.append(df)
    except Exception as e:
        print(f"❌ Error reading {f}: {e}")

if dfs:
    combined = pd.concat(dfs, ignore_index=True)
    out_path = folder / "all_result.csv"
    combined.to_csv(out_path, index=False)
    print(f"✅ Combined CSV saved to {out_path}")
else:
    print("⚠️ No CSV files found in the folder.")