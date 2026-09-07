Telecom Customer Analytics and Monthly Charges Prediction
Etokhana Mercy 
SheCode Africa Data Analysis & Machine Learning Capstone

1. Problem Statement
Telecom providers need to understand what drives a customer’s MonthlyCharges in order to design fair, competitive pricing and service bundles. This project analyzes the Telco Customer Churn dataset to identify which customer characteristics and subscribed services are associated with higher or lower monthly charges, then builds a Linear Regression model to predict MonthlyCharges from those characteristics.
2. Dataset
Source: Telco Customer Churn from Kaggle platform (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Rows: 7,043 customers, Target: MonthlyCharges
Predictors used: tenure, SeniorCitizen, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod.
 3. Data Cleaning
TotalCharges was stored as text due to blank values for new customers (`tenure == 0`); converted to numeric and set to 0 for those rows.
Checked for duplicate rows and missing values and performed imputation where necessary.
TotalCharges was excluded from modeling it is derived from tenure × MonthlyCharges, so including it would leak the target into the features.
4. Exploratory Data Analysis
Distribution of MonthlyCharges: a large cluster near $20 (phone-only customers) and a broader hump between ~$65–$100 (customers with internet + add-ons).
Tenure vs. MonthlyCharges (scatter, corr.) only a weak relationship; tenure reflects loyalty, not spend level.
Monthly Charges by Contract (boxplot) One-year and Two-year contract customers have higher, tighter-clustered charges than Month-to-month.
Monthly Charges by Internet Service (boxplot) Fiber optic customers pay noticeably more (median ~$90) than DSL (~$55); customers with no internet service cluster tightly near $20.
Categorical variable spread: InternetService produces the largest gap in mean MonthlyCharges of any categorical variable, ahead of individual add-on services.
Correlation matrix of numerical variables: tenure and TotalCharges are strongly correlated (as expected, since TotalCharges accumulates over tenure); SeniorCitizen and tenure show only weak correlation with MonthlyCharges on their own.
Actual vs. Predicted MonthlyCharges: (test set) points fall almost exactly on the diagonal, indicating a very close fit.
Residual plot: residuals are small (roughly ±4) and show no strong trend across the range of predictions, consistent with a well-specified linear relationship.
5. Methodology
Encoding: categorical variables one-hot encoded via a ColumnTransformer (passthrough for numeric columns, OneHotEncoder for categorical columns), wrapped in an sklearn Pipeline with the LinearRegression regressor.
Split: 80/20 train/test split large enough test set (1,400 rows) to evaluate generalization while retaining most data for training.
Model: sklearn.linear_model.LinearRegression, trained via the pipeline on the training set.
6. Model Evaluation
Metrics and values
MAE: 0.789
MSE: 1.105 
RMSE: 1.051 
R²: 0.9988
The model’s predictions are off by less than $1 on average (MAE $0.79), and it explains 99.9% of the variance in MonthlyCharges. MonthlyCharges is generated as an (approximately) additive function of which services a customer subscribes to, not a noisy real-world billing outcome. Because the true relationship is close to linear in the service flags, Linear Regression is a strong structural fit, not an overfit the actual-vs-predicted plot (near-perfect diagonal) and the residual plot (small, patternless residuals) both support this.
Top coefficient: InternetService = Fiber optic carries the largest positive coefficient (+15.21), confirming Fiber optic subscription as the single strongest driver of higher MonthlyCharges, holding another services constant.
7. Key Insights
InternetService is the dominant driver of MonthlyCharges. Fiber optic customers pay substantially more than DSL customers; customers with no internet service pay the least.
Add-on services compound the effect Streaming TV/Movies, Device Protection, OnlineBackup/Security and Tech Support each add to the base charge.
Tenure and SeniorCitizen status barely affect price on their own what a customer pays is set by their service mix, not how long they have been a customer or their age bracket.
Contract type differences are secondary to service mix: One-year/Two-year customers charge higher mainly because they tend to have fuller service bundles, not because of the contract length itself.
MonthlyCharges is close to a linear (additive) function of subscribed services in this dataset, which is why Linear Regression fits it so well (R² ≈ 0.999).
8. Recommendations
Adopt clearer bundle tiers (e.g., Fiber Essentials vs. Fiber Complete) given how much
Fiber optic + add-ons drives charges this could reduce pricing confusion and support upsell.
Target DSL customers without add-ons for security/backup upsell a segment with clear room to grow revenue without a full fiber upgrade.
Re-examine month-to-month pricing and incentives, since these customers show the most variability in charges and (per general churn literature on this dataset) the highest churn risk.
Additional data that would improve future prediction: usage volume (data/minutes), regional pricing tier, promotional discount flags, and subscription start dates for add-ons.
9. Limitations
Linear Regression assumes linear, additive relationships; while that fits this dataset well, it would not necessarily generalize to a provider with genuinely non-linear or bundled pricing.
The very high R² reflects the structure of this particular (semi-synthetic) dataset and should not be read as a universal benchmark for monthly-charge prediction accuracy.
Single-market snapshot coefficients reflect this provider’s specific pricing structure.
10. Citation
Dataset: BlastChar, *Telco Customer Churn*, Kaggle.
https://www.kaggle.com/datasets/blastchar/telco-customer-churn
