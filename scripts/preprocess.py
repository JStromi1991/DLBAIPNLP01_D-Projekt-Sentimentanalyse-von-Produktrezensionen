"""
Schritt 5: Textvorverarbeitung fuer die Sentimentanalyse.

Enthaelt die Vorverarbeitungsfunktionen (Tokenisierung, Stoppwortentfernung,
Stemming), die anschliessend in ein TF-IDF-Vektorisierungsschema
(numerische Codierung) einfliessen. Wird von train_and_evaluate.py importiert.

Bewusste Designentscheidung: Es werden ausschliesslich Werkzeuge verwendet,
die ohne zusaetzlichen Download externer Sprachressourcen funktionieren
(reine Regex-Tokenisierung, sklearns eingebaute Stoppwortliste, NLTKs
regelbasierter Porter-Stemmer), damit das Skript ueberall reproduzierbar
laeuft.
"""

import re

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_PATTERN = re.compile(r"[a-zA-Z]+")
_stemmer = PorterStemmer()
_STOPWORDS = ENGLISH_STOP_WORDS


def tokenize_and_stem(text: str) -> list[str]:
    """Vorverarbeitungspipeline fuer einen einzelnen Text:
    1) Kleinschreibung, 2) Tokenisierung (nur alphabetische Tokens),
    3) Stoppwortentfernung, 4) Stemming (Porter-Stemmer)."""
    text = text.lower()
    tokens = _TOKEN_PATTERN.findall(text)
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    stemmed = [_stemmer.stem(t) for t in tokens]
    return stemmed


def combine_title_text(df):
    return (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)


if __name__ == "__main__":
    # kleiner Selbsttest
    sample = "The Games were AMAZING! I really loved playing these games with my friends."
    print(tokenize_and_stem(sample))
