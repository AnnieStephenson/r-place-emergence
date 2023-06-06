import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os
import pickle
import rplacem.variables_rplace2022 as var

# Prepare input data as numpy arrays
file_path = os.path.join(var.DATA_PATH, 'training_data.pickle')
with open(file_path, 'r') as f:
    [inputvals, outputval] = pickle.load(f)

#X = np.random.randint(0, high=20, size=(1000,4))
#y = np.mean(X, axis=1)
print(inputvals, inputvals.shape)
print(outputval, outputval.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(inputvals, outputval, test_size=0.2, random_state=2)

# Set XGBoost regressor parameters
params = {
    'objective': 'reg:squarederror',  # Regression task with squared loss
    'max_depth': 3,                    # Maximum depth of each tree
    'learning_rate': 0.1,              # Learning rate
    'subsample': 0.8,                  # Subsample ratio of the training instances
    'colsample_bytree': 0.8,           # Subsample ratio of features
    'random_state': 42                 # Random seed
}

num_rounds = 500  # Number of boosting rounds (iterations)

# Train the model
model = XGBRegressor(**params, n_estimators=num_rounds)
model.fit(X_train, y_train)

# Evaluate the model on the testing set
predictions = model.predict(X_test)
print(X_test)
relerr =  (predictions-np.mean(X_test, axis=1)) / np.mean(X_test, axis=1)
print(relerr)
print(np.mean(np.abs(relerr)))