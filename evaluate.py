import pandas as pd
import numpy as np

# 1. LOAD DATASET (Ganti nama CSV misal udah di-unzip)
DATA_PATH = "dataset.csv"

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"[+] Loaded {len(df)} records dari {path}.")
        return df
    except FileNotFoundError:
        print(f"[!] Data {path} belom ada. Lupa wget/kaggle pull?")
        return pd.DataFrame()

def analyze_benchmarks(df: pd.DataFrame):
    if df.empty: return
    
    # Asumsi ada kolom 'model' dan skor macam 'MMLU', 'GSM8K', 'params_b'
    print("\n--- SAMPLE RAW METRICS ---")
    print(df.head(3))
    
    # Custom aggregations/mechanics (contoh aja)
    print("\n--- MATRIX EVALUATION (WIP) ---")
    # ... logic lu taruh sini (e.g. cosine sim, pivot, top 10 models)
    pass

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    analyze_benchmarks(df)
