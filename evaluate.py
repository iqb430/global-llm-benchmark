import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = "dataset.csv"

def print_brutalist_header(text):
    print(f"\n[{text.upper()}]\n{'='*50}")

def load_data(path: str) -> pd.DataFrame:
    '''Load dataset murni, buang row NaN buat regresi'''
    df = pd.read_csv(path)
    return df.dropna(subset=['context_window', 'api_input_cost_per_1k_usd', 'open_source', 'mmlu_score'])

def analyze_and_predict(df: pd.DataFrame):
    # 1. CORE INSIGHTS (HARD DATA)
    print_brutalist_header("Global LLM Matrix: Raw Data")
    print(f"Total Models       : {len(df)}")
    print(f"Open Source        : {df['open_source'].sum()}")
    print(f"Proprietary        : {len(df) - df['open_source'].sum()}")

    # 2. PREDICTIVE ENGINEERING (ML PIPELINE)
    # Target: mmlu_score
    # Features: context_window, input_cost, open_source (bool to int)
    
    df['open_source_int'] = df['open_source'].astype(int)
    X = df[['context_window', 'api_input_cost_per_1k_usd', 'open_source_int']]
    y = df['mmlu_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    # 3. METRICS EVALUATION
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    
    print_brutalist_header("Machine Learning Analytics (Random Forest)")
    print(f"Target Feature     : MMLU Score")
    print(f"Train/Test Split   : 80/20")
    print(f"R-Squared (R2)     : {r2:.4f} (Accuracy variance)")
    print(f"RMSE               : {rmse:.4f} (Error margin)")
    
    # Feature Importance
    importances = rf_model.feature_importances_
    print("\n[FEATURE IMPORTANCE]")
    for feature, imp in zip(X.columns, importances):
        print(f"{feature:<25} : {imp:.4f}")

    # 4. PRAGMATIC RECOMMENDATIONS
    print_brutalist_header("Cost-Cognition Efficiency")
    # Cari model paling over-performing dengan cost terendah
    df['efficiency_ratio'] = df['mmlu_score'] / ((df['api_input_cost_per_1k_usd'] + df['api_output_cost_per_1k_usd'])/2 + 1e-6)
    
    top_efficient = df.nlargest(3, 'efficiency_ratio')
    print("Top 3 Elite Performers / Free (Self-Hosted):")
    for _, row in top_efficient.iterrows():
        print(f"-> {row['model_name']:<15} | Provider: {row['provider']:<10} | MMLU: {row['mmlu_score']} | Efisiensi: Sangat Tinggi")

    print("\n[SYSTEM] Halted. Evaluasi Mekanik Selesai.")

if __name__ == "__main__":
    df = load_data(DATA_PATH)
    analyze_and_predict(df)
