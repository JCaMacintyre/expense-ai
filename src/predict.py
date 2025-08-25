import pandas as pd

# very simple keyword → category mapping
RULES = {
    "STARBUCKS": "Coffee",
    "UBER": "Transport",
    "LYFT": "Transport",
    "SHELL": "Gas",
    "CHEVRON": "Gas",
    "AMAZON": "Shopping",
    "WHOLE FOODS": "Groceries",
    "WALMART": "Groceries",
    "NETFLIX": "Entertainment",
    "SPOTIFY": "Entertainment",
    "CHIPOTLE": "Dining",
    "PAYROLL": "Income",
}

def apply_rules(description: str) -> str:
    if not isinstance(description, str):
        return "Other"
    text = description.upper()
    for keyword, category in RULES.items():
        if keyword in text:
            return category
    return "Other"

def categorize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'rule_category' column based on keyword matching."""
    df["rule_category"] = df["description_clean"].map(apply_rules)
    return df
