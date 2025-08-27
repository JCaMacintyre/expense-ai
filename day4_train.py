from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

DATA = Path("data/transactions_labeled_final.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True, parents=True)

def main():
    if not DATA.exists():
        raise FileNotFoundError(f"Missing {DATA}.")
    df = pd.read_csv(DATA)

    # Expect these columns
    for col in ["description_clean", "manual_category"]:
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {DATA}. Found: {list(df.columns)}")

    # Filter out unlabeled rows
    df = df[df["manual_category"].astype(str).str.len() > 0].copy()
    if len(df) < 100:
        raise ValueError(f"Not enough labeled rows: {len(df)} (need ~200 to start).")

    X = df["description_clean"].fillna("")
    y = df["manual_category"].astype(str)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Strong baseline: TF-IDF + Logistic Regression
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), min_df=2)),
        ("clf", LogisticRegression(max_iter=200))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"\nAccuracy: {acc:.3f}\n")
    print(classification_report(y_test, preds))

    # Save model pipeline (vectorizer + classifier together)
    model_path = MODEL_DIR / "expense_category_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model -> {model_path}")

    # Optional: save confusion matrix
    cm = confusion_matrix(y_test, preds, labels=sorted(y.unique()))
    cm_path = MODEL_DIR / "confusion_matrix.csv"
    pd.DataFrame(cm, index=sorted(y.unique()), columns=sorted(y.unique())).to_csv(cm_path)
    print(f"Saved confusion matrix -> {cm_path}")

if __name__ == "__main__":
    main()
