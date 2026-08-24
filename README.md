# Machine Learning Projects

A collection of supervised and unsupervised machine learning projects completed as part of a
machine learning course, using Python and scikit-learn. Each project covers the full workflow:
data cleaning and preprocessing, modeling, evaluation, and interpretation of results.

## Tools & Libraries
Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

---

## 1. Finding Donors for CharityML (Supervised Learning)

**Problem:** A nonprofit wants to identify individuals likely to earn more than $50,000 per year —
a useful proxy for donation capacity — so it can target its outreach more efficiently. This is a
binary classification problem using 1994 U.S. Census data (~45,000 records).

**Approach:**
- Preprocessed the data: log-transformed skewed features, normalized numerical features, and
  one-hot encoded categorical features (103 features after encoding).
- Established a naive-predictor baseline for comparison.
- Trained and compared three classifiers (Logistic Regression, Random Forest, AdaBoost) across
  1%, 10%, and 100% training-set sizes, measuring accuracy, F-score, and training/prediction time.
- Selected AdaBoost as the best model and tuned it with GridSearchCV.
- Extracted feature importances and evaluated performance on a reduced feature set.

**Result:** The tuned AdaBoost model achieved ~86% accuracy and a ~0.74 F-score on the test set,
far outperforming the naive baseline (~25% accuracy) and demonstrating that the model learned
meaningful patterns from the data.

**Files:** `finding_donors.ipynb`

---

## 2. Identify Customer Segments (Unsupervised Learning)

**Problem:** A German mail-order company wants to understand which parts of the general population
form its core customer base. Using real demographic data for the general population (~891,000
people) and the company's customers (~191,000 people), the goal is to segment the population and
compare segment representation between the two groups.

**Approach:**
- Performed extensive data cleaning: re-encoded missing-value codes as NaN, dropped high-missing
  columns, split off high-missing rows, engineered mixed-type features, and re-encoded categoricals —
  all packaged into a reusable cleaning function.
- Applied imputation and feature scaling, then used PCA for dimensionality reduction (retaining 30
  components capturing ~89% of variance) and interpreted the leading principal components.
- Clustered the general population with K-Means (10 clusters, chosen via the elbow method), then
  applied the same fitted transformations to the customer data.
- Compared cluster proportions between the two groups to identify over- and under-represented
  segments, programmatically selecting the segments with the largest differences.

**Result:** Identified a settled, lower-density suburban segment as the company's core customer base,
and a dense, urban, multi-family-housing segment as strongly under-represented among customers.

**Note on data:** The demographic datasets for this project are proprietary (provided by Bertelsmann
Arvato) and are **not included** in this repository per the data-use agreement. Only the notebook is
provided.

**Files:** `Identify_Customer_Segments.ipynb`
