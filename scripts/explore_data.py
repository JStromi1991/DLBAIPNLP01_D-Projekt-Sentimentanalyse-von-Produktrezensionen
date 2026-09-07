"""
Schritt 4: Daten laden, zusammenfuehren und explorative Statistiken
fuer den Projektbericht (Sentimentanalyse von Produktrezensionen) erstellen.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CATEGORIES = ["Video_Games", "Electronics", "Pet_Supplies"]


def load_all() -> pd.DataFrame:
    frames = []
    for cat in CATEGORIES:
        path = DATA_DIR / f"reviews_{cat}.csv"
        df = pd.read_csv(path)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    return full


def main() -> None:
    df = load_all()
    print(f"Gesamtanzahl geladener Rezensionen: {len(df)}")

    # --- Datenqualitaet pruefen ---
    n_missing = df[["title", "text", "rating"]].isna().sum()
    n_duplicates = df.duplicated(subset=["title", "text", "category"]).sum()
    print("\nFehlende Werte pro Spalte:")
    print(n_missing)
    print(f"\nDuplikate (gleicher Titel+Text+Kategorie): {n_duplicates}")

    # Duplikate und leere Texte entfernen (Datenbereinigung)
    before = len(df)
    df = df.dropna(subset=["title", "text", "rating"])
    df = df.drop_duplicates(subset=["title", "text", "category"])
    df["rating"] = df["rating"].astype(int)
    after = len(df)
    print(f"\nNach Bereinigung: {after} von {before} Rezensionen behalten.")

    # --- Textlaenge (in Woertern) ---
    df["n_words"] = (df["title"] + " " + df["text"]).str.split().str.len()

    # --- Tabelle 1: Anzahl Rezensionen pro Kategorie ---
    count_table = df.groupby("category").size().rename("n_reviews")
    print("\nAnzahl Rezensionen pro Kategorie:")
    print(count_table)

    # --- Tabelle 2: Verteilung der Bewertungen (1-5) pro Kategorie ---
    rating_dist = (
        df.groupby(["category", "rating"]).size().unstack(fill_value=0)
    )
    rating_dist_pct = rating_dist.div(rating_dist.sum(axis=1), axis=0) * 100
    print("\nVerteilung der Bewertungen pro Kategorie (absolut):")
    print(rating_dist)
    print("\nVerteilung der Bewertungen pro Kategorie (in %):")
    print(rating_dist_pct.round(1))

    # --- Textlaengen-Statistik ---
    length_stats = df.groupby("category")["n_words"].agg(
        ["mean", "median", "min", "max"]
    ).round(1)
    print("\nTextlaenge (Woerter, Titel+Beschreibung) pro Kategorie:")
    print(length_stats)

    # --- Train/Test-Split jetzt schon anlegen (fuer Tabelle in Kap. 3) ---
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[["category", "rating"]].astype(str).agg("_".join, axis=1)
    )
    print(f"\nTrainingssatz: {len(train_df)} Rezensionen")
    print(f"Testsatz: {len(test_df)} Rezensionen")

    split_counts = pd.DataFrame({
        "train": train_df.groupby("category").size(),
        "test": test_df.groupby("category").size(),
    })
    print("\nAnzahl Rezensionen im Trainings-/Testsatz je Kategorie:")
    print(split_counts)

    # --- Alles speichern ---
    df.to_csv(DATA_DIR / "reviews_all_clean.csv", index=False)
    train_df.to_csv(DATA_DIR / "reviews_train.csv", index=False)
    test_df.to_csv(DATA_DIR / "reviews_test.csv", index=False)

    count_table.to_csv(RESULTS_DIR / "table_counts_per_category.csv")
    rating_dist.to_csv(RESULTS_DIR / "table_rating_distribution_abs.csv")
    rating_dist_pct.round(1).to_csv(RESULTS_DIR / "table_rating_distribution_pct.csv")
    length_stats.to_csv(RESULTS_DIR / "table_text_length_stats.csv")
    split_counts.to_csv(RESULTS_DIR / "table_train_test_split.csv")

    print("\nAlle Tabellen wurden im Ordner 'results' gespeichert.")


if __name__ == "__main__":
    main()
