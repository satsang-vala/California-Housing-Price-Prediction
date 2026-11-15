import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_att , cat_att):

    num_pipeline = Pipeline([
        ("imputer" , SimpleImputer(strategy="median")),
        ("scaler" , StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("encode" , OneHotEncoder(handle_unknown= "ignore")),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline , num_att),
        ("cat" , cat_pipeline , cat_att)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")
    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0.0 , 1.5 , 3.0 , 4.5 , 6.0 , np.inf],
                                   labels=[1,2,3,4,5])
    split = StratifiedShuffleSplit(n_splits= 1 , test_size= 0.2 , random_state= 42)

    for train_index , test_index in split.split(housing , housing['income_cat']):
        housing.loc[test_index].drop("income_cat" , axis =1).to_csv("input.csv" , index = False)
        housing_train = housing.loc[train_index].drop("income_cat" , axis = 1)

    housing_labels = housing_train["median_house_value"].copy()
    housing_features = housing_train.drop("median_house_value" , axis =1)

    num_att = housing_features.drop("ocean_proximity" , axis =1).columns.to_list()
    cat_att = ["ocean_proximity"]

    pipeline = build_pipeline(num_att, cat_att)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state= 42)
    model.fit(housing_prepared , housing_labels)

    joblib.dump(model , MODEL_FILE)
    joblib.dump(pipeline , PIPELINE_FILE)
    print("congrats! , you trained the model")

else:
    model =joblib.load("model.pkl")
    pipeline =joblib.load("pipeline.pkl")

    input_data = pd.read_csv("input.csv")
    input_x = input_data.drop("median_house_value" , axis =1)
    actual_data = input_data["median_house_value"].copy()

    transformed_input = pipeline.transform(input_x)
    predictions = model.predict(transformed_input)

    input_data["predicted_median_house_value"] = predictions

    # input_data.to_csv("output.csv", index=False)
    input_data.to_csv("output_new.csv", index=False)


    from sklearn.metrics import mean_absolute_error

    mae = mean_absolute_error(actual_data, predictions)
    print("Mean Absolute Error:", mae)

    from sklearn.metrics import root_mean_squared_error

    rmse = root_mean_squared_error(actual_data, predictions)
    print("RMSE:", rmse)


