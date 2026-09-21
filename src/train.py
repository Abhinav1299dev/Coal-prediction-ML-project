import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from preprocess import load_data, preprocess_data, split_data

# Load + preprocess
df = load_data()
df = preprocess_data(df)

# Split
X, y = split_data(df)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

# Hyperparameters
n_estimators = [int(x) for x in np.linspace(100, 1200, 12)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, 6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}

# Model
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    scoring='neg_mean_squared_error',
    n_iter=10,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Train
rf_random.fit(X_train, y_train)

# Save model
pickle.dump(rf_random, open("model/model.pkl", "wb"))

print("✅ Model trained and saved!") 

# -------------------------------
# 👉 ADD ROC CURVE CODE BELOW
# -------------------------------

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Predict probabilities (IMPORTANT)
y_probs = rf_random.predict_proba(X_test)[:, 1]

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Compute AUC
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')

# Save figure
plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')

print("ROC curve saveed as roc_curve.png")
