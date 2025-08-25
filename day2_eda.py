from src.preprocess import load_transactions
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/transactions_mock.csv")  # change this if you use a different file
OUTPUT_PATH = Path("data/transactions_clean.csv")

def main():
    print("Loading:", DATA_PATH)
    df = load_transactions(str(DATA_PATH))

    # Basic checks
    print("\n=== HEAD (first 5 rows) ===")
    print(df.head())

    print("\n=== SHAPE (rows, columns) ===")
    print(df.shape)

    print("\n=== COLUMNS ===")
    print(df.columns.tolist())

    # Count missing values per column
    print("\n=== MISSING VALUES PER COLUMN ===")
    print(df.isna().sum())

    # Quick spend summary
    expenses = df[df["amount"] < 0]["amount"].sum() if "amount" in df else 0
    income   = df[df["amount"] > 0]["amount"].sum() if "amount" in df else 0
    print("\n=== SPEND SUMMARY ===")
    print(f"Total expenses: {expenses:.2f}")
    print(f"Total income:   {income:.2f}")
    print(f"Net:            {income + expenses:.2f}")

    # Save cleaned copy
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned CSV -> {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()
