# Sentimentanalyse von Produktrezensionen

Projektbericht DLBAIPNLP01_D – Projekt: NLP (Aufgabenstellung 1)

## Kurzbeschreibung

Ein Mehrklassen-Textklassifikationssystem, das Amazon-Produktrezensionen
(Titel + Beschreibung) einer numerischen Bewertung von 1 bis 5 zuordnet.
Verwendete Kategorien: **Video_Games, Electronics, Pet_Supplies**
(je ca. 5.000 Rezensionen, Datensatz: McAuley Lab, Amazon Reviews'23).

## Projektstruktur

```
data/       Rohdaten (reviews_<Kategorie>.csv), bereinigte Gesamtdaten
            sowie Trainings-/Testsplit
scripts/
  fetch_reviews_local.py   Sammelt die Rohdaten (lokal ausfuehren, s.u.)
  explore_data.py          Datenbereinigung, Statistiken, Train/Test-Split
  preprocess.py            Textvorverarbeitung (Tokenisierung, Stoppwort-
                            entfernung, Stemming)
  train_and_evaluate.py    TF-IDF-Vektorisierung, Training (Logistic
                            Regression), Evaluation
results/    Ergebnistabellen (CSV/JSON) fuer den Bericht
figures/    Confusion-Matrix-Plot
```

## Ausfuehren

```bash
pip install -r requirements.txt

# 1. Rohdaten sammeln (nur einmal, benoetigt Internetzugang)
python scripts/fetch_reviews_local.py

# 2. Datenaufbereitung & Explorative Statistiken
python scripts/explore_data.py

# 3. Modelltraining & Evaluation
cd scripts && python train_and_evaluate.py
```

## Methodik (Kurzfassung)

- **Vorverarbeitung:** Kleinschreibung, Regex-Tokenisierung (nur
  alphabetische Tokens), Entfernen englischer Stoppwoerter
  (scikit-learn-Stoppwortliste), Stemming (Porter-Stemmer).
- **Numerische Codierung:** TF-IDF (Unigramme + Bigramme, max. 20.000
  Merkmale, min_df=3).
- **Modell:** Logistic Regression (multinomial, C=5.0, max_iter=1000).
- **Split:** 80 % Training / 20 % Test, stratifiziert nach
  Kategorie x Bewertung (random_state=42).

## Ergebnisse (Testsatz)

Siehe `results/metrics_summary.json`, `results/confusion_matrix.csv` und
`figures/confusion_matrix.png`.
