# Business Problem  
Loan defaults increase credit risk and consequently reduce financial institutions' earnings. The goal of this project is to develop a solution to improve credit risk classification methods and identify the key indicators that help predict a customer's probability of default.

Future versions of this project will focus on:

* Retrain the model using GridSearchCV to explore a wider range of hyperparameter combinations. This step has not been performed yet due to the computational cost involved and its relatively low priority given the scope of the initial business problem;
* Automate credit analysis: Identify relevant credit and customer characteristics to support credit decisions and propose adjustments to reduce risk, leveraging RAG (Retrieval-Augmented Generation) and AI Agents;
* Risk-based interest rate pricing: Develop a pricing approach based on the customer's risk rating (probability of default), credit spread, operational costs, and the prevailing interest rate environment.

# Dataset
This analysis uses the credit-risk-dataset from Kaggle (https://www.kaggle.com/datasets/laotse/credit-risk-dataset/data), which contains 32,581 rows and 12 features.
* person_age: Age (int64)
* person_income: Annual Income (int64)
* person_home_ownership: Home ownership (object)
* person_emp_length: Employment length (in years) (float64)
* loan_intent: Loan intent (object)
* loan_grade: Loan grade (object)
* loan_amnt: Loan amount (int64)
* loan_int_rate: Interest rate (float64)
* loan_status: Loan status (0 = non default, 1 = default) (int64)
* loan_percent_income: Percent income (float64)
* cb_person_default_on_file: Historical default (object)
* cb_preson_cred_hist_length: Credit history length (int64)

This project focuses on predicting loan_status, with particular emphasis on identifying default cases (loan_status == 1).

# Exploratory Data Analysis
The exploratory data analysis revealed missing values and outliers that required further investigation and treatment.
	* person_emp_length: 895 missing values,
	* loan_int_rate: 3,116 missing values,
	* person_age: max 144 years,
	* person_emp_length: max 123 years.

Key findings of the exploratory analysis:

* There are 6 types of Credit Products and a balanced loan volume (distribution/ volume)
	* EDUCATION (19.58% - 61.2M),
	* MEDICAL (18.00% - 56.2M),
	* VENTURE (17.54% - 54.8M),
	* PERSONAL (16.91% - 52.8M),
	* DEBTCONSOLIDATION (16.01% - 50.0M),
	* HOMEIMPROVEMENT (11.96% - 37.4M).
	* Total portfolio = 312,320,100

* When looking into the credit portfolio risk ratings, 18.06% are graded in D+ ratings (high risk), representing 56,402,625.
	* D: 12.6% - 39.3M,
	* E: 4.0% - 12.5M,
	* F: 1.1% - 3.6M,
	* G: 0.4% - 1.1M.

* About interest rates pricing, we found (volume/ weighted average interest rate):
	* HOMEIMPROVEMENT (volume: 37.4 mi - 11.59%) and 
	* MEDICAL (volume: 56.2 mi - 11.36%).

* The highest default rates per product impacts mostly:
	* DEBTCONSOLIDATION  (1.490 operations - default volume: 16.6M - 33.1% of total in DEBTCONSOLIDATION), 
	* MEDICAL (1.621 operations - default volume: 18.2M - 32.3% of total in MEDICAL) and 
	* HOMEIMPROVEMENT: (941 operations - default volume: 9.3 mi - 24.9% of total in HOMEIMPROVEMENT).

* In ratings the default proportion looks like this:
	* D: 57,9% default rate - 22.8M,
	* E: 62,9% default rate- 7.8M,
	* F: 70,4% default rate- 2.5M,
	* G: 99,9% default rate- 1.1M.
	* Total D+ default = 34,187,825 (60,6% of D+ portfolio and 11% of total portfolio)

* Interest rates and loan-to-income ratio react along loan grades (as risk grows, pricing and DSTI increase):
	* D: interest rate: 15,4% - loan-to-income ratio: 23.1%,
	* E: interest rate: 17,0% - loan-to-income ratio: 24.0%,
	* F: interest rate: 18,5% - loan-to-income ratio: 25.3%,
	* G: interest rate: 20,2% - loan-to-income ratio: 27.1%.

* Finally, when it comes to understanding default cases (loan_status == 1) the most correlated features are: 
	* loan_percent_income (DSTI)
	* person_home_ownership (Home Ownership)
	* cb_person_default_on_file (Credit Bureau default)

# Data Preprocessing
Several important preprocessing steps were performed to prepare the dataset for Machine Learning:

* Target variable definition: Separated the target variable (loan_status, y) from the predictor variables (X).
* Train-test split: Divided the dataset into training (80%) and test (20%) sets, using stratification to preserve the target variable distribution across both sets.
* Outlier treatment: Removed outliers from person_age (maximum value of 100) and person_emp_length (maximum value of 50).
* Missing value imputation: Handled missing values using SimpleImputer, applying the median strategy to numerical features and the most frequent value strategy to categorical features.
* Feature scaling: Applied StandardScaler to numerical features to standardize their scale and reduce the impact of differences in magnitude.
* Categorical encoding: Applied OneHotEncoder to convert categorical features into numerical representations.

These preprocessing steps ensure that the dataset is properly formatted and ready for use by the selected Machine Learning algorithms.

# Models
Several Machine Learning algorithms were implemented and evaluated to identify the best-performing model for this classification problem: Logistic Regression, Decision Tree, Random Forest, Extra Trees, Gradient Boosting and XGBoost.

# Hyperparameter Optimization
Hyperparameter optimization was performed using RandomizedSearchCV with 5-fold cross-validation.

This approach randomly samples different hyperparameter combinations and evaluates their performance across multiple validation folds, helping identify promising configurations while reducing computational costs compared to an exhaustive grid search.

The model definitions and hyperparameter search spaces are available in 2_modeling.ipynb, under the section "Models definition and hyperparameters".

# Model Evaluation
The best performing model achieved the following evaluation metrics:
* CV_PR_AUC: 0.899997
* Accuracy: 0.934479
* Precision: 0.965388
* Recall: 0.725738
* F1: 0.828583
* ROC_AUC: 0.951593
* PR_AUC: 0.907118
* Balanced_Accuracy: 0.859238
* Training_Time_Seconds: 4.65

Consfusion Matrix
* Non Defaulters (TN = 5058/ FN = 37)
* Defaulters (FP = 390/ TP = 1032)

# Results and Conclusions
Best-performing model, limitations and business implications.

With a total training time of 4,65 seconds, and PR_AUC of 0.90 in cross-validation, the best-performing model was XGBoost with hiperparameters:
{
'model__subsample': 0.85, 
'model__reg_lambda': 1, 
'model__reg_alpha': 0, 
'model__n_estimators': 300, 
'model__min_child_weight': 7, 
'model__max_depth': 5, 
'model__learning_rate': 0.1, 
'model__gamma': 0.5, 
'model__colsample_bytree': 0.7
}

The top 3 most important variables during traing were:
* loan_percent_income: 28,14%;
* person_home_ownership: 19,35%;
* person_income: 16,61%;
* 64% of relevance in total.

The model delivered good predictive performance with a relatively low computational cost. However, its Recall score for default cases was 0.73, meaning it failed to identify approximately 27% of actual defaults, while achieving 0.99 Recall for non-default cases. Therefore, despite its overall performance, I would not recommend deploying it in a production environment at this stage, as undetected defaults could represent a significant financial risk.

Further hyperparameter tuning, threshold optimization, and access to more comprehensive data (e.g., collateral information, payment history, and debt obligations) could improve its predictive capabilities. Overall, XGBoost provides a promising starting point for credit risk classification, but further validation and business-oriented testing are necessary before production deployment.
