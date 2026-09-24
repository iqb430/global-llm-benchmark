import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# === BRUTALIST HUD ===
class HUD:
    CYAN = '\033[96m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    @staticmethod
    def header(text):
        print(f"\n{HUD.RED}[::] {text.upper()} [::]{HUD.RESET}\n{HUD.CYAN}{'='*60}{HUD.RESET}")

def engineer_features(df: pd.DataFrame) -> tuple:
    HUD.header("Phase 1: Feature Engineering & Mechanics")
    initial_len = len(df)
    
    df = df.dropna(subset=['mmlu_score', 'context_window']).copy()
    
    df['log_context'] = np.log1p(df['context_window'])
    df['avg_cost'] = (df['api_input_cost_per_1k_usd'] + df['api_output_cost_per_1k_usd']) / 2
    df['is_open_source'] = df['open_source'].astype(int)
    
    df['cognitive_density'] = df['mmlu_score'] / np.log1p(df['context_window'])
    df['cost_penalty'] = np.where(df['avg_cost'] > 0, 1 / (df['avg_cost'] + 1e-6), 1e6) 
    
    features = ['log_context', 'avg_cost', 'is_open_source', 'cognitive_density']
    X = df[features]
    y = df['mmlu_score']
    
    print(f"Data Purged      : {initial_len - len(df)} ghost rows deleted.")
    print(f"Feature Space    : {X.shape[1]} dimensional matrix constructed.")
    return X, y, df

def train_ensemble_engine(X, y):
    HUD.header("Phase 2: Ensemble ML Engine (Cross-Validation)")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        'XGBoost': xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42)
    }
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print(f"{'MODEL':<20} | {'CV R2 SCORE (MEAN)':<20} | {'VOLATILITY (STD)':<15}")
    print("-" * 60)
    
    best_model_name = ""
    best_score = -float('inf')
    best_model = None
    
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
        mean_score = scores.mean()
        std_score = scores.std()
        print(f"{HUD.YELLOW}{name:<20}{HUD.RESET} | {mean_score:>18.4f} | {std_score:>15.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_model_name = name
            best_model = model
            
    print(f"\n[*] Apex Engine Selected: {best_model_name} (Accuracy: {best_score:.4f})")
    
    best_model.fit(X_scaled, y)
    
    HUD.header("Phase 3: Structural Subconscious (Feature Weights)")
    importances = best_model.feature_importances_
    for col, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 30)
        print(f"{col:<20} | {imp:.4f} | {HUD.CYAN}{bar}{HUD.RESET}")
    
def map_anomalies(df):
    HUD.header("Phase 4: Anomaly Detection (Value Outliers)")
    
    df['raw_value'] = (df['mmlu_score'] * df['log_context']) / (df['avg_cost'] + 1e-4)
    outliers = df.nlargest(3, 'raw_value')
    
    print("Top 3 Market Disruptors (High Cognition, Zero/Low Cost):")
    for _, row in outliers.iterrows():
        cost_str = "FREE" if row['avg_cost'] == 0 else f"${row['avg_cost']:.4f}"
        print(f"[-] {row['model_name']:<15} (Provider: {row['provider']:<10}) -> MMLU: {row['mmlu_score']:<5} | Cost: {cost_str}")

if __name__ == "__main__":
    import os
    os.system('clear')
    df = pd.read_csv("dataset.csv")
    X, y, clean_df = engineer_features(df)
    train_ensemble_engine(X, y)
    map_anomalies(clean_df)
    print(f"\n{HUD.RED}[SYSTEM HALTED]{HUD.RESET}\n")
