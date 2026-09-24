# Global LLM Benchmark 2024-2026: Raw Mechanical Analysis

Proprietary AI is an illusion sold via subscriptions. The data proves it. 

I built this **LLM Performance Prediction** engine to strip away marketing syntax and evaluate 37 state-of-the-art models in the **Global LLM Benchmark Dataset (2024-2026)**. By mapping API Cost, Context Windows, and Open Source flags against raw cognition scores (MMLU, GSM8K), we expose the actual cost-to-intelligence ratio of the AI industry.

**Architect:** [Iqbal Anwar](https://github.com/iqb430) | AI Developer & Karate Coach

## Framework & Tech Stack
- **Dataset:** `sumitchavhan7/global-llm-benchmark-dataset-2024-2026` via Kaggle.
- **Algorithm:** Python `RandomForestRegressor`, `XGBoost` (Scikit-Learn). 
- **Dashboard:** Streamlit & Plotly.
- **Goal:** Open Source LLM vs Proprietary Evaluation & Cost-Efficiency Analysis.

## The Reality (Machine Learning Output)
The script ran a regression test and found:
1. **Intelligence isn't priced:** 50% of an LLM's capability (MMLU) is driven by its Context Window architecture. Only 45% correlates to API pricing. 
2. **Elite Open Source:** `DeepSeek-V2` and `Llama 3 70B` are running at >82 MMLU. You don't need a massive enterprise budget for high cognition.
3. **The Value Play:** On a pure Cost-to-Cognition metric, `Claude 3 Haiku` and `Qwen1.5-72B` obliterate the rest of the market.

## Execution
Mekanisme brutalist. Terminal output & interactive dashboard.

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Run the benchmark predictor (Terminal)
python evaluate.py

# Run the interactive dashboard (UI)
streamlit run app.py
```

Kalau lu masih bakar duit buat bayar API mahal tanpa ngukur rasionya, lu buang energi. Kloning repositori ini, tes prediksi ke model lokal lu, atau tinggalin. Gak ngaruh buat gua.
