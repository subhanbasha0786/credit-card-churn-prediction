# Credit Card Customer Churn: Analysis and Prediction

A bank wants to know which credit card customers are about to leave, and why. This project explores 10,127 customers, finds the behaviours linked to churn, and builds machine learning models to flag at-risk customers.

## Business Questions
1. How many customers churn, and who are they?
2. Which behaviours separate churners from loyal customers?
3. Can a model predict churn and find the customers who really leave?
4. What should the bank do about it?

## Dataset
- Kaggle: Credit Card Customers (BankChurners.csv), 10,127 customers, 21 columns
- Target: `Attrition_Flag` (Attrited Customer = churned): 1,627 churned (16.1%), 8,500 existing
- Features: demographics, card type, products held, contacts with the bank, inactivity, credit limit, revolving balance, transaction count and amount, Q4 vs Q1 change in activity
- No missing values or duplicates. `Unknown` categories were kept as their own group.

## Key Insights
1. **16.1% of customers churn.**
2. **Behaviour beats demographics.** Age, gender, income and education move churn only slightly (for example 17.4% for women vs 14.6% for men).
3. **Low activity is the strongest signal.** Customers with 60 or fewer transactions in the year churn at 31 to 37%, while those with 61 to 80 churn at 5.8%, and those with more than 80 churn at under 2%.
4. **Zero revolving balance means high risk.** 36.2% of customers with no revolving balance churn, vs 9.6% of those with a balance.
5. **Fewer products means higher churn.** Customers with 1 or 2 products churn at 26 to 28%, compared with 10 to 12% for customers with 4 to 6 products.
6. **Many contacts with the bank is a warning sign.** Churn climbs from 1.8% (0 contacts) to 33.5% (5 contacts). All 54 customers with 6 contacts churned (small group).

## Model Results
Test set: 2,026 customers (325 churners). Metrics are for the churn class.

| Model               | Accuracy | Precision | Recall | F1   | ROC-AUC | PR-AUC |
|---------------------|----------|-----------|--------|------|---------|--------|
| Logistic Regression | 85.6%    | 53.3%     | 81.9%  | 0.65 | 0.921   | 0.731  |
| Random Forest       | 94.9%    | 93.0%     | 73.5%  | 0.82 | 0.987   | 0.944  |

- **Random Forest is the better model.** It caught 239 of 325 churners and wrongly flagged only 18 loyal customers. 5-fold cross-validated AUC is 0.988, so the result is stable.
- **Logistic Regression catches more churners (266 of 325)** but wrongly flags 233 loyal customers, so it would waste retention effort.
- Most important features (Random Forest): total transaction amount, total transaction count, revolving balance, and change in transaction count from Q1 to Q4.

## Business View: Who To Contact First
Ranking test customers by Random Forest risk score:

| Customers contacted | Churners caught | % of all churners caught | Hit rate |
|---------------------|-----------------|--------------------------|----------|
| Top 10% (202)       | 195             | 60.0%                    | 96.5%    |
| Top 20% (405)       | 309             | 95.1%                    | 76.3%    |
| Top 30% (607)       | 321             | 98.8%                    | 52.9%    |

Calling only the highest-risk 20% of customers would reach about 95% of the customers who are going to leave.

## Recommendations
- Score customers every month and contact the highest-risk group first.
- Trigger a retention offer when a customer's transactions drop sharply versus the previous quarter.
- Cross-sell a second or third product to single-product customers.
- Review customers who contacted the bank many times; they may have unresolved complaints.

## Limitations
- The activity features cover the same year in which customers left, so the model partly spots customers who are already leaving. A real early-warning system should use activity from earlier months only.
- These findings are associations from historical data, not proof of cause. The model should be tested on new data before real use.

## Files
- `churn_analysis.ipynb`: full analysis and models (run in Google Colab)
- `model_results.csv`, `risk_ranking.csv`: result tables
- `screenshots/`: charts (demographics, behaviour, churners vs existing, correlations, model evaluation, feature importance)

## How to Run
1. Download BankChurners.csv from the Kaggle dataset "Credit Card Customers"
2. Open `churn_analysis.ipynb` in Google Colab
3. Run all cells and upload the CSV when asked

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Google Colab

## Author
Subhan | https://linkedin.com/in/subhan-basha-proddutur-36686725a
