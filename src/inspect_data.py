import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

files_to_check = [
    "interactions.tsv",
    "pathbank_all_proteins.csv",
    "pathbank_pathways.csv",
]


def inspect_file(filename):
  filepath = os.path.join(DATA_DIR, filename)
  print("=" * 70)
  print(f"DOSYA: {filename}")

  if filename.endswith(".tsv"):
    # İlk 2 yorum satırını atla
    df = pd.read_csv(filepath, sep="\t", skiprows=2, nrows=5, low_memory=False)
  else:
    df = pd.read_csv(filepath, nrows=5, low_memory=False)

  print(f"Sütunlar ({len(df.columns)} adet): {list(df.columns)}")
  print("İlk 2 satır:")
  print(df.head(2).to_string())


if __name__ == "__main__":
  for fname in files_to_check:
    inspect_file(fname)