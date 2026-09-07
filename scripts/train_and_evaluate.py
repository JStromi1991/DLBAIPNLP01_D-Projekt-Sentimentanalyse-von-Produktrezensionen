"""
Schritte 6-8: Numerische Codierung (TF-IDF), Modelltraining
(Logistic Regression) und Evaluation (Accuracy, Confusion Matrix).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from preprocess import combine_title_text, tokenize_and_stem

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = BASE_DIR / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42


def main() -> None:
    train_df = pd.read_csv(DATA_DIR / "reviews_train.csv")
    test_df = pd.read_csv(DATA_DIR / "reviews_test.csv")

    X_train_text = combine_title_text(train_df)
    X_test_text = combine_title_text(test_df)
    y_train = train_df["rating"].astype(int)
    y_test = test_df["rating"].astype(int)

    # --- Numerische Codierung: TF-IDF mit eigener Vorverarbeitung ---
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize_and_stem,
        preprocessor=lambda x: x,   # Vorverarbeitung erledigt tokenize_and_stem
        lowercase=False,            # bereits in tokenize_and_stem enthalten
        max_features=20000,
        min_df=3,
        ngram_range=(1, 2),
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    print(f"TF-IDF Merkmalsraum: {X_train.shape[1]} Terme")

    # --- Modelltraining: Logistic Regression (Mehrklassen) ---
    model = LogisticRegression(
        max_iter=1000,
        C=5.0,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nGesamt-Accuracy auf dem Testsatz: {acc:.4f}")

    # Majority-Baseline zum Vergleich (immer die haeufigste Klasse vorhersagen)
    majority_class = y_train.mode()[0]
    baseline_acc = (y_test == majority_class).mean()
    print(f"Majority-Baseline (immer '{majority_class}' vorhersagen): {baseline_acc:.4f}")

    report = classification_report(y_test, y_pred, digits=3)
    print("\nClassification Report:\n", report)

    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5])
    cm_df = pd.DataFrame(cm, index=[f"wahr_{i}" for i in range(1, 6)],
                          columns=[f"pred_{i}" for i in range(1, 6)])
    print("\nVerwirrungsmatrix:\n", cm_df)

    # Accuracy pro Produktkategorie (optionale Zusatzanalyse)
    test_df = test_df.copy()
    test_df["y_pred"] = y_pred
    per_cat_acc = (
        test_df.assign(correct=lambda d: d["rating"] == d["y_pred"])
        .groupby("category")["correct"].mean()
        .rename("accuracy")
    )
    print("\nAccuracy je Produktkategorie:\n", per_cat_acc)

    # --- Ergebnisse speichern ---
    cm_df.to_csv(RESULTS_DIR / "confusion_matrix.csv")
    per_cat_acc.to_csv(RESULTS_DIR / "accuracy_per_category.csv")

    with open(RESULTS_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "n_features": int(X_train.shape[1]),
            "accuracy": float(acc),
            "majority_baseline_accuracy": float(baseline_acc),
            "majority_class": int(majority_class),
        }, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    # --- Confusion-Matrix-Plot ---
    fig, ax = plt.subplots(figsize=(5, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[1, 2, 3, 4, 5])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix (Logistic Regression)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    print(f"\nConfusion-Matrix-Plot gespeichert unter {FIG_DIR / 'confusion_matrix.png'}")

    print("\nAlle Ergebnisse wurden im Ordner 'results' gespeichert.")


if __name__ == "__main__":
    main()
