from src.preprocess import load_transactions
from src.predict import categorize_dataframe
from pathlib import Path

INPUT = Path("data/transactions_clean.csv")
OUTPUT = Path("data/transactions_labeled.csv")

def main():
    print("Loading cleaned data...")
    df = load_transactions(str(INPUT))
    print("Rows:", len(df))

    # apply rule-based categorization
    df = categorize_dataframe(df)

    print("\n=== Sample with categories ===")
    print(df[["date", "description", "amount", "rule_category"]].head(10))

    df.to_csv(OUTPUT, index=False)
    print(f"\nSaved labeled file -> {OUTPUT.resolve()}")

if __name__ == "__main__":
    main()
