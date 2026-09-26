"""
Enterprise-grade LLM Benchmarker
"""
import logging
import os
import sys
from typing import Tuple, Dict, Any
import warnings

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# === BRUTALIST HUD ===
class HUD:
    CYAN = '\033[96m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    @staticmethod
    def header(text: str) -> None:
        print(f"\n{HUD.RED}[::] {text.upper()} [::]{HUD.RESET}\n{HUD.CYAN}{'='*60}{HUD.RESET}")


class Benchmarker:
    """Evaluates and ranks LLMs based on performance and cost metrics."""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.models: Dict[str, Any] = {
            'XGBoost': xgb.XGBRegressor(n_estimators=150, max_depth=4, learning_rate=0.05, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=150, random_state=42),
            'Random Forest': RandomForestRegressor(n_estimators=150, random_state=42)
        }
        self.scaler = StandardScaler()
        self.best_model_name = ""
        self.best_model = None

    def load_data(self) -> pd.DataFrame:
        """Loads dataset from the specified path."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            return pd.read_csv(self.data_path)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        """Engineers features and returns processed X, y arrays along with the cleaned df."""
        HUD.header("Phase 1: Feature Engineering & Mechanics")
        initial_len = len(df)
        
        try:
            # Drop rows without targets
            df = df.dropna(subset=['mmlu_score', 'context_window']).copy()
            
            # Non-Linear Feature Construction
            df['log_context'] = np.log1p(df['context_window'])
            df['avg_cost'] = (df['api_input_cost_per_1k_usd'] + df['api_output_cost_per_1k_usd']) / 2
            df['is_open_source'] = df['open_source'].astype(int)
            
            # Efficiency matrix: penalize high cost, heavily reward context+score
            df['cognitive_density'] = df['mmlu_score'] / np.log1p(df['context_window'])
            df['cost_penalty'] = np.where(df['avg_cost'] > 0, 1 / (df['avg_cost'] + 1e-6), 1e6) 
            
            features = ['log_context', 'avg_cost', 'is_open_source', 'cognitive_density']
            X = df[features]
            y = df['mmlu_score']
            
            logger.info(f"Data Purged: {initial_len - len(df)} rows deleted.")
            logger.info(f"Feature Space: {X.shape[1]} dimensional matrix constructed.")
            return X, y, df
        except KeyError as e:
            logger.error(f"Missing expected column in dataset: {e}")
            raise

    def train_ensemble_engine(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Evaluates ensemble models using cross-validation and extracts feature importance."""
        HUD.header("Phase 2: Ensemble ML Engine (Cross-Validation)")
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            
            print(f"{'MODEL':<20} | {'CV R2 SCORE (MEAN)':<20} | {'VOLATILITY (STD)':<15}")
            print("-" * 60)
            
            best_score = -float('inf')
            
            for name, model in self.models.items():
                scores = cross_val_score(model, X_scaled, y, cv=kf, scoring='r2')
                mean_score = scores.mean()
                std_score = scores.std()
                print(f"{HUD.YELLOW}{name:<20}{HUD.RESET} | {mean_score:>18.4f} | {std_score:>15.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    self.best_model_name = name
                    self.best_model = model
                    
            print(f"\n[*] Apex Engine Selected: {self.best_model_name} (Accuracy: {best_score:.4f})")
            
            if self.best_model is not None:
                self.best_model.fit(X_scaled, y)
            
            HUD.header("Phase 3: Structural Subconscious (Feature Weights)")
            if self.best_model is not None:
                importances = self.best_model.feature_importances_
                for col, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
                    bar = "█" * int(imp * 30)
                    print(f"{col:<20} | {imp:.4f} | {HUD.CYAN}{bar}{HUD.RESET}")
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            raise

    def map_anomalies(self, df: pd.DataFrame) -> None:
        """Identifies anomalies or top disruptive models based on performance and cost."""
        HUD.header("Phase 4: Anomaly Detection (Value Outliers)")
        try:
            # Normalized efficiency score purely for mathematical ranking
            df['raw_value'] = (df['mmlu_score'] * df['log_context']) / (df['avg_cost'] + 1e-4)
            outliers = df.nlargest(3, 'raw_value')
            
            print("Top 3 Market Disruptors (High Cognition, Zero/Low Cost):")
            for _, row in outliers.iterrows():
                cost_str = "FREE" if row['avg_cost'] == 0 else f"${row['avg_cost']:.4f}"
                print(f"[-] {row['model_name']:<15} (Provider: {row['provider']:<10}) -> MMLU: {row['mmlu_score']:<5} | Cost: {cost_str}")
        except KeyError as e:
            logger.error(f"Missing column for anomaly detection: {e}")
            raise

    def run(self) -> None:
        """Executes the full benchmarking pipeline."""
        df = self.load_data()
        X, y, clean_df = self.engineer_features(df)
        self.train_ensemble_engine(X, y)
        self.map_anomalies(clean_df)

if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "dataset.csv")
    benchmarker = Benchmarker(dataset_path)
    try:
        benchmarker.run()
        print(f"\n{HUD.RED}[SYSTEM HALTED]{HUD.RESET}\n")
    except Exception as e:
        logger.critical(f"Benchmarking pipeline failed: {e}")
        sys.exit(1)
