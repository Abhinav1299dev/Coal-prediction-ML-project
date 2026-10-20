
from pathlib import Path

import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = Path("data") / "Coal Mines Dataset india.csv"
TARGET_COLUMN = "Coal/ Lignite Production"


# =========================================================
# LOAD DATASET
# =========================================================

def main():
    if not DATA_PATH.exists():
        print(f"Error: Dataset not found at '{DATA_PATH}'.")
        return

    try:
        df = pd.read_csv(DATA_PATH)

        if TARGET_COLUMN not in df.columns:
            print(f"Error: Column '{TARGET_COLUMN}' not found.")
            print("Available columns:", df.columns.tolist())
            return

        # Convert production values to numeric where possible.
        production = pd.to_numeric(
            df[TARGET_COLUMN], errors="coerce"
        ).dropna()

        if production.empty:
            print("Error: No valid production values found.")
            return

        # -------------------------------------------------
        # PRODUCTION STATISTICS
        # -------------------------------------------------

        print("=" * 48)
        print("       COAL PRODUCTION DATA ANALYSIS")
        print("=" * 48)

        print(f"Dataset           : {DATA_PATH.name}")
        print(f"Total records     : {len(df):,}")
        print(f"Valid observations: {len(production):,}")
        print(f"Missing/invalid   : {len(df) - len(production):,}")

        print("-" * 48)
        print("PRODUCTION STATISTICS")
        print("-" * 48)

        print(f"Minimum Production: {production.min():,.2f}")
        print(f"Maximum Production: {production.max():,.2f}")
        print(f"Average Production: {production.mean():,.2f}")
        print(f"Median Production : {production.median():,.2f}")

        print("=" * 48)
        print("Analysis completed successfully.")

    except (OSError, pd.errors.ParserError) as error:
        print(f"Error reading dataset: {error}")


if __name__ == "__main__":
    main()
