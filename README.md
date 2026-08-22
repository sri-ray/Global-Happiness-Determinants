# Structural Determinants of Global Happiness Scores

An empirical machine learning project analyzing international developmental and macroeconomic indicators to predict national happiness scores and classify high- vs. low-happiness regimes.

---

### 📌 Project Context
* **Institution:** University of Zurich
* **Seminar:** Big Data Methods for Economists Seminar (Co-authored)
* **Target Variable:** World Happiness Report `Ladder score` (continuous regression target and median-split binary regime)
* **Feature Space:** 40+ World Bank economic, governance, health, and labor market indicators

---

### 🛠 My Core Contributions & Methodology
I developed the automated regularized feature selection and classification pipelines in Python (`scikit-learn`):

1. **High-Dimensional Feature Normalization:** Standardized feature space using `StandardScaler` to prevent shrinkage distortion across scale-variant metrics (e.g., GDP per capita vs. percentage indicators).
2. **10-Fold Cross-Validated LASSO:** Tuned the optimal penalty ($\alpha \approx 0.0118$) to eliminate 13 redundant indicators, isolating 27 non-zero predictors with $R^2 = 0.7235$ and $\text{RMSE} = 0.5675$.
3. **Multicollinearity Screening (VIF):** Implemented an iterative Variance Inflation Factor loop ($\text{threshold} \le 5.0$), pruning highly collinear indicators (e.g., agricultural employment, vulnerable employment) down to 23 stable features.
4. **Recursive Feature Elimination (RFE) & Logistic Classification:** Isolated the top 10 structural tipping points and estimated an $L_2$-regularized logistic model (achieving 95% accuracy and an AUC of 0.99).
5. **Economic Interpretation:** Extracted coefficient odds ratios to quantify key drivers (e.g., governance accountability, government consumption, and health expenditure dynamics).

---

### 🧰 Tech Stack
* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`
* **Environment:** Jupyter Notebook

---

### 📂 Repository Structure
* `data/Dataset.csv`: Extracted indicator dataset with World Bank and happiness metrics.
* `notebooks/happiness_analysis.ipynb`: Complete data cleaning, LASSO tuning, VIF filtering, and classification pipeline.

---

### 🚀 How to Run

1. **Install dependencies:**
    ```bash
   pip install pandas numpy scikit-learn matplotlib
   ```
2. Open `notebooks/happiness_analysis.ipynb` in Jupyter Notebook or VS Code.
3. Execute cells sequentially to reproduce the cross-validation curves, confusion matrix, and ROC evaluation.
