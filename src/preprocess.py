import pandas as pd
import re

EXPECTED_COLS = ["date", "description", "amount"]

def clean_description(text: str) -> str:
    """Normalize vendor text to help matching later."""
    if not isinstance(text, str):
        return ""
    t = text.upper()
    t = re.sub(r"\s+", " ", t)          # collapse spaces
    t = re.sub(r"[^A-Z0-9 &/.\-]", "", t)  # keep simple chars
    return t.strip()

def load_transactions(path: str) -> pd.DataFrame:
    """
    Load a CSV and try to standardize column names to: date, description, amount.
    Returns a DataFrame with:
      - date (datetime64 if possible)
      - description (original)
      - amount (float)
      - description_clean (uppercase normalized)
    """
    df = pd.read_csv(path)

    # Make a mapping from lowercase names to actual names
    lower_map = {c.lower().strip(): c for c in df.columns}

    # Try to find likely columns
    date_col = lower_map.get("date") or lower_map.get("posted") or lower_map.get("transaction date")
    desc_col = lower_map.get("description") or lower_map.get("merchant") or lower_map.get("details")
    amt_col  = lower_map.get("amount") or lower_map.get("amt") or lower_map.get("value")

    # Fallbacks if something's missing
    if desc_col is None:
        # pick first column as description-ish
        desc_col = df.columns[0]
    if amt_col is None:
        # pick second column as amount-ish
        amt_col = df.columns[1]

    # Rename to standard names when found
    rename_map = {}
    if date_col: rename_map[date_col] = "date"
    if desc_col: rename_map[desc_col] = "description"
    if amt_col:  rename_map[amt_col]  = "amount"
    df = df.rename(columns=rename_map)

    # Parse types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Clean vendor text
    if "description" in df.columns:
        df["description_clean"] = df["description"].map(clean_description)
    else:
        df["description"] = ""
        df["description_clean"] = ""

    # Drop totally empty rows and reset index
    df = df.dropna(how="all").reset_index(drop=True)

    return df
