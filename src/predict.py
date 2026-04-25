import pickle
from preprocess import load_data, preprocess_data, split_data

# Load model
model = pickle.load(open("model/model.pkl", "rb"))

# Load and preprocess data
df = load_data()
df = preprocess_data(df)

X, y = split_data(df)

# Take one sample from dataset
sample = X.iloc[0].values.reshape(1, -1)

# Predict
result = model.predict(sample)

print("Prediction:", result) 
