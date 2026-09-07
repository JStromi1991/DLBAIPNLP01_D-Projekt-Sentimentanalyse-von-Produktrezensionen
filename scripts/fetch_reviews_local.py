"""
Sammelt eine kompakte Stichprobe aus dem Amazon-Reviews-2023-Datensatz
(McAuley Lab, 2023) fuer drei Produktkategorien.

WICHTIG: Dieses Skript muss LOKAL (z. B. in Google Colab oder auf dem
eigenen Rechner) ausgefuehrt werden, NICHT in der Cloud-Sandbox von Claude -
von dort ist der Zugriff auf externe Datenserver aus Sicherheitsgruenden
gesperrt.

Vorgehen:
1. Dieses Skript in Google Colab hochladen (oder lokal mit
   `pip install requests` ausfuehren).
2. Ausfuehren -> erzeugt fuer jede Kategorie eine CSV-Datei
   (z. B. reviews_Video_Games.csv) mit ca. 5000 Rezensionen.
3. Die drei erzeugten CSV-Dateien wieder im Chat mit Claude hochladen.

Hinweis: Es werden nur die ersten ~50.000 Zeilen jeder (sehr grossen,
unkomprimierten .jsonl-) Rohdaten-Datei gestreamt (nicht komplett
heruntergeladen), daraus wird eine Zufallsstichprobe von SAMPLE_SIZE
Rezensionen gezogen. Das haelt den Download klein und schnell, auch fuer
grosse Kategorien wie "Electronics".
"""

import csv
import json
import random
import time
import urllib.request

CATEGORIES = ["Video_Games", "Electronics", "Pet_Supplies"]
SAMPLE_SIZE = 5000        # Zielanzahl Rezensionen pro Kategorie
MAX_LINES_TO_SCAN = 50000  # wie viele Zeilen der Rohdatei max. gelesen werden
SEED = 42

# Hinweis: Die Dateien liegen unkomprimiert als .jsonl vor (kein .gz).
URL_TEMPLATE = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/{category}.jsonl"
)

random.seed(SEED)


def reservoir_sample_reviews(category: str) -> list[dict]:
    """Streamt die jsonl-Datei einer Kategorie und zieht per
    Reservoir-Sampling eine Zufallsstichprobe, ohne die komplette
    (oft mehrere GB grosse) Datei herunterzuladen."""
    url = URL_TEMPLATE.format(category=category)
    print(f"[{category}] Starte Stream von: {url}")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    reservoir: list[dict] = []
    n_seen = 0

    with urllib.request.urlopen(req, timeout=60) as response:
        for raw_line in response:
            if n_seen >= MAX_LINES_TO_SCAN:
                break
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                continue

            text = (record.get("text") or "").strip()
            title = (record.get("title") or "").strip()
            rating = record.get("rating")
            if not text or rating is None:
                continue

            n_seen += 1
            item = {
                "category": category,
                "title": title,
                "text": text,
                "rating": int(rating),
            }

            if len(reservoir) < SAMPLE_SIZE:
                reservoir.append(item)
            else:
                j = random.randint(0, n_seen - 1)
                if j < SAMPLE_SIZE:
                    reservoir[j] = item

            if n_seen % 5000 == 0:
                print(f"[{category}] ... {n_seen} Zeilen gelesen")

    print(f"[{category}] fertig: {len(reservoir)} Rezensionen gesammelt "
          f"(aus {n_seen} gelesenen Zeilen)")
    return reservoir


def main() -> None:
    for category in CATEGORIES:
        t0 = time.time()
        reviews = reservoir_sample_reviews(category)
        out_path = f"reviews_{category}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["category", "title", "text", "rating"]
            )
            writer.writeheader()
            writer.writerows(reviews)
        dt = time.time() - t0
        print(f"[{category}] gespeichert unter {out_path} ({dt:.1f}s)\n")

    print("Fertig! Bitte die erzeugten CSV-Dateien "
          "(reviews_Video_Games.csv, reviews_Electronics.csv, "
          "reviews_Pet_Supplies.csv) im Chat hochladen.")


if __name__ == "__main__":
    main()
