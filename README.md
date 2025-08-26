# Bank Marketing Analysis and Dashboard

This project analyzes the "Bank Marketing" dataset to identify key factors influencing whether a customer subscribes to a term deposit. It includes a detailed exploratory data analysis (EDA) in a Jupyter notebook and an interactive dashboard built with Streamlit.

## Dataset

The dataset used is `bank-full.csv`, which contains information about bank marketing campaigns. The data includes customer demographics, campaign details, and the outcome of the campaign (whether the customer subscribed to a term deposit).

## Installation

To set up the environment and install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

There are two main components to this project: the analysis notebook and the interactive dashboard.

### 1. Exploratory Data Analysis (Jupyter Notebook)

The `analysis_fixed.ipynb` notebook contains a comprehensive analysis of the dataset. To run the notebook, you will need to have Jupyter Notebook or JupyterLab installed.

```bash
jupyter notebook analysis_fixed.ipynb
```

The notebook covers:
- Data loading and initial inspection
- Data quality checks (missing values)
- Exploratory data analysis with visualizations:
    - Distribution of the target variable
    - Correlation matrix of numeric features
    - Analysis of categorical features and their impact on conversion rates
- A baseline Logistic Regression model to predict subscription.

### 2. Interactive Dashboard

The `dashboard.py` script launches a Streamlit dashboard that provides a high-level overview of the data and model performance.

To run the dashboard, execute the following command in your terminal:

```bash
streamlit run dashboard.py
```

The dashboard features:
- Key Performance Indicators (KPIs) such as total campaigns, subscription numbers, and conversion rate.
- Visualizations of class balance, numeric feature correlations, and conversion rates by key categorical features.
- A baseline Logistic Regression model with its ROC-AUC score and ROC curve.

## Analysis and Findings

The analysis reveals several key insights into the factors that influence a customer's decision to subscribe to a term deposit. The most significant predictors include:

- **Positive Predictors:**
    - Previous campaign outcome (`poutcome_success`)
    - Contacting customers in specific months (e.g., March, October, September)
    - Customer's job (e.g., retired, student)

- **Negative Predictors:**
    - Lack of previous contact (`poutcome_unknown`)
    - Certain housing and loan statuses
    - Contacting customers in other months (e.g., July, November, August)

The baseline Logistic Regression model achieves an accuracy of approximately 90% and an ROC-AUC score of 0.90, providing a good starting point for further model improvements.

## Files in the Project

- `analysis_fixed.ipynb`: Jupyter Notebook with the detailed data analysis.
- `bank-full.csv`: The dataset used for the analysis.
- `dashboard.py`: The Streamlit dashboard application.
- `requirements.txt`: A list of Python dependencies for the project.
- `README.md`: This file.
