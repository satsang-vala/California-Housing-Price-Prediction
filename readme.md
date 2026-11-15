
# 🏡 California Housing Price Prediction

## 📌 Project Overview

This project focuses on predicting median house prices in California using Machine Learning. It uses the California Housing dataset and a Random Forest Regression model to estimate house prices based on different housing and demographic features.

The main objective is to implement an end-to-end Machine Learning workflow, including data preprocessing, feature engineering, model training, prediction, and evaluation.

## 🎯 Project Objectives

- Explore and understand housing data.
- Handle missing values in numerical features.
- Apply feature scaling and categorical encoding.
- Split data into training and testing sets using stratified sampling.
- Build reusable preprocessing pipelines using Scikit-learn.
- Train a Random Forest Regression model.
- Save and load trained models using Joblib.
- Evaluate model performance using MAE and RMSE.

## 📂 Project Structure

```text
California-Housing-Price-Prediction/
│
├── prediction_model.py  # Main ML training and prediction code
├── requirements.txt     # Required Python libraries
├── .gitignore           # Files excluded from Git tracking
└── README.md            # Project documentation
```

## 📊 Dataset Information

The project uses the California Housing dataset containing 20,640 records and 10 columns.

The features include:

| Feature | Description |
|---|---|
| longitude | Longitude of the location |
| latitude | Latitude of the location |
| housing_median_age | Median age of houses |
| total_rooms | Total number of rooms |
| total_bedrooms | Total number of bedrooms |
| population | Population of the area |
| households | Number of households |
| median_income | Median income of the area |
| median_house_value | Median house value (Target) |
| ocean_proximity | Location relative to the ocean |

**Target Variable:** `median_house_value`

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib

## ⚙️ Machine Learning Workflow

### 1. Data Preprocessing

- Loaded the housing dataset using Pandas.
- Separated input features and target variable.
- Identified numerical and categorical attributes.

### 2. Stratified Train-Test Split

Used `StratifiedShuffleSplit` to divide the dataset into training and testing sets.

Created income categories using `pd.cut()` based on the `median_income` feature to maintain a representative income distribution in both datasets.

- Training data: 80%
- Testing data: 20%

### 3. Data Transformation Pipeline

Created separate preprocessing pipelines for numerical and categorical features.

**Numerical Pipeline:**
- `SimpleImputer(strategy="median")` – Handles missing values.
- `StandardScaler()` – Scales numerical features.

**Categorical Pipeline:**
- `OneHotEncoder(handle_unknown="ignore")` – Converts categorical features into numerical representations.

Combined both pipelines using `ColumnTransformer`.

### 4. Model Training

Used `RandomForestRegressor` from Scikit-learn to train the model on the preprocessed housing data.

### 5. Model Persistence

Used Joblib to save the trained model and preprocessing pipeline:

- `model.pkl`
- `pipeline.pkl`

This allows the trained model and preprocessing steps to be reused without training the model again.

### 6. Prediction

Loaded the saved model and preprocessing pipeline to transform the test dataset and generate predicted median house values.

The predictions are saved in `output_new.csv`.

## 📈 Model Evaluation

Evaluated the model using two regression metrics:

**Mean Absolute Error (MAE)**

Measures the average absolute difference between actual and predicted house prices.

**Root Mean Squared Error (RMSE)**

Measures the square root of the average squared prediction errors, giving greater weight to larger errors.

## 🚀 How to Run This Project

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/California-Housing-Price-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd California-Housing-Price-Prediction
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Download or obtain the California Housing dataset and place `housing.csv` in the project directory.

### 5. Run the Python File

```bash
python prediction_model.py
```

**Note:** The script checks whether `model.pkl` exists.

- If the model does not exist, it trains a new Random Forest model and saves the model and preprocessing pipeline.
- If the model already exists, it loads the saved model and generates predictions using `input.csv`.

Make sure the required dataset files are available locally before running the script.

## 📚 Key Learnings

Through this project, I gained practical experience with:

- Building end-to-end Machine Learning workflows.
- Understanding numerical and categorical data preprocessing.
- Creating reusable Scikit-learn pipelines.
- Applying stratified sampling.
- Understanding Random Forest Regression.
- Saving and loading Machine Learning models.
- Making predictions on unseen test data.
- Evaluating regression models using MAE and RMSE.

## 🔮 Future Improvements

- Perform Exploratory Data Analysis (EDA) with meaningful visualizations.
- Experiment with different regression algorithms.
- Perform hyperparameter tuning.
- Improve prediction accuracy.
- Build a simple web application using Flask or Streamlit.
- Deploy the trained model.

## 👨‍💻 Author

**Satsang Vala**

Computer Engineering Graduate | Aspiring Data Scientist

- GitHub: https://github.com/satsang-vala
- LinkedIn: Add your LinkedIn profile here

---

⭐ If you find this project useful, feel free to star the repository!
