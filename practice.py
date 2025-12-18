import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import numpy as np
train_data, test_data = train_test_split(pd.read_csv('battery_health.csv'),
                                          test_size=0.2, random_state=42)

attrs = ["model_year", "daily_usage_hours", "charging_cycles",
          "battery_health_percent","avg_charge_limit_percent"]

# scatter_matrix(train_data[attrs])
# plt.show()

train_data_labels = train_data['battery_health_percent']
train_data = train_data.drop("battery_health_percent", axis=1)

cat_attrs = ["brand", "os", "usage_type","overheating_issues"]
num_attrs = ["model_year", "daily_usage_hours", "charging_cycles",
             "avg_charge_limit_percent", "battery_age_months","performance_rating"]

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder())
])
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attrs),
    ("cat", cat_pipeline, cat_attrs)
])
train_data_prepared = full_pipeline.fit_transform(train_data)

lin_reg = LinearRegression()
lin_reg.fit(train_data_prepared, train_data_labels)

first_five_data = train_data.iloc[:5]
first_five_data_prep = full_pipeline.transform(first_five_data)

first_five_predictions = lin_reg.predict(first_five_data_prep)
first_five_labels = list(train_data_labels.iloc[:5])

print("predicts for first 5 laptops: ",  first_five_predictions)
print("actual labels: ", first_five_labels)

full_train_predictions = lin_reg.predict(train_data_prepared)

mse = mean_squared_error(train_data_labels, full_train_predictions)
rmse = np.sqrt(mse)

print("mean squared error", mse)
print("root mean squared error ", rmse)


# test data
test_labels = test_data["battery_health_percent"]
test_data = test_data.drop("battery_health_percent", axis=1)

test_data_prep = full_pipeline.transform(test_data)

test_predictions = lin_reg.predict(test_data_prep)

test_result_dict = {
    'predictions': test_predictions,
    'labels': test_labels
}
test_result_table = pd.DataFrame(test_result_dict)
print(test_result_table)

mse = mean_squared_error(test_labels, test_predictions)
rmse = np.sqrt(mse)

print("mse for test set: ", mse)
print("rmse for test set: ", rmse)


