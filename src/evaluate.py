import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from preprocess import load_data, preprocess_data, split_data
from sklearn.model_selection import train_test_split
import numpy as np

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# Load data
df = load_data()
df = preprocess_data(df)

X, y = split_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

predict = model.predict(X_test)

# Error plot
plt.figure(figsize=(13,8))
sns.histplot(y_test - predict, kde=True, color="maroon")
plt.axvline(0, color='black', linestyle='--')
plt.title("Error Distribution")

# Regression plot
plt.figure(figsize=(20,8))
sns.regplot(x=y_test, y=predict, color='teal')
plt.title("Actual vs Predicted")

# Metrics
print("MAE:", metrics.mean_absolute_error(y_test, predict))
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, predict)))

plt.show()
