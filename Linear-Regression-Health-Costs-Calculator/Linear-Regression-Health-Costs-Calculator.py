# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure to import keras correctly from TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Step 1: Download and load the dataset
# Downloading the 'insurance.csv' file
!wget -q https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv
dataset = pd.read_csv('insurance.csv')

# Display the first few rows to confirm the data is loaded
print(dataset.head())

# Step 2: Encoding categorical columns (run this part only once)
# Check if 'region' column exists and encode the categorical columns
if 'region' in dataset.columns:
    # Map 'sex' column values to 0 and 1
    dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
    # Map 'smoker' column values to 0 and 1
    dataset['smoker'] = dataset['smoker'].map({'yes': 1, 'no': 0})
    # Convert the 'region' column into dummy variables (one-hot encoding)
    dataset = pd.get_dummies(dataset, columns=['region'], drop_first=True)
else:
    print("Categorical columns have already been processed.")

# Step 3: Separate the 'expenses' column from the features
labels = dataset.pop('expenses')  # 'expenses' will be our label (target)

# Step 4: Split the data into training and testing sets
# We are using an 80-20 split for training and testing datasets
train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    dataset, labels, test_size=0.2, random_state=42
)

# Step 5: Normalize numeric features (standardize)
scaler = StandardScaler()
numeric_columns = ['age', 'bmi', 'children']
# Apply scaling to the training set
train_dataset[numeric_columns] = scaler.fit_transform(train_dataset[numeric_columns])
# Apply the same scaling to the test set
test_dataset[numeric_columns] = scaler.transform(test_dataset[numeric_columns])

# Verify the transformed datasets
print(train_dataset.head())
print(test_dataset.head())

# Step 6: Verify the columns of the dataset after transformations
# Print column names to ensure proper transformations
print("Columns of the dataset after transformations:", dataset.columns)

# If 'expenses' is still a column in the dataset, separate it out
if 'expenses' in dataset.columns:
    labels = dataset.pop('expenses')  # Remove 'expenses' and assign it to labels
else:
    print("Error: The 'expenses' column is not present in the dataset.")

# Now that we have the labels, let's split the labels into training and testing sets
train_labels, test_labels = train_test_split(labels, test_size=0.2, random_state=42)

# Verify that the labels are correctly separated
print(f"First few rows of train_labels:\n{train_labels.head()}")
print(f"First few rows of test_labels:\n{test_labels.head()}")

# Step 7: Normalize the dataset again (scaling the entire training and test data)
scaler = StandardScaler()
train_dataset_scaled = scaler.fit_transform(train_dataset)  # Scaling the training dataset
test_dataset_scaled = scaler.transform(test_dataset)  # Scaling the test dataset

# Verify the scaled data
print(f"First few rows of train_dataset_scaled:\n{train_dataset_scaled[:5]}")
print(f"First few rows of test_dataset_scaled:\n{test_dataset_scaled[:5]}")

# Step 8: Create the model with fewer regularization layers
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[train_dataset_scaled.shape[1]]),  # Input layer
    layers.Dense(128, activation='relu'),  # Hidden layer
    layers.Dense(1)  # Output layer (predicting a single value)
])

# Step 9: Compile the model with an adjusted learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae', 'mse'])

# Step 10: Implement Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Step 11: Train the model with more epochs and EarlyStopping
history = model.fit(train_dataset_scaled, train_labels, epochs=200, validation_split=0.2, verbose=2,
                    callbacks=[early_stopping])

# Step 12: Evaluate the model on the test set
loss, mae, mse = model.evaluate(test_dataset_scaled, test_labels, verbose=2)
print(f"Mean Absolute Error on test set: {mae:.2f}")
print(f"Mean Squared Error on test set: {mse:.2f}")

# Step 13: Visualize the predictions and residuals (errors)
import matplotlib.pyplot as plt

# Step 14: Get predictions from the model on the test set
predictions = model.predict(test_dataset_scaled)

# Step 15: Plot the predictions vs the real values
plt.figure(figsize=(10,6))

# Scatter plot of predictions vs actual values
plt.subplot(1, 2, 1)
plt.scatter(test_labels, predictions)
plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], color='red', lw=2)  # Identity line
plt.title('Predictions vs Actual Values')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.grid(True)

# Step 16: Plot the distribution of the errors (residuals)
errors = predictions.flatten() - test_labels

plt.subplot(1, 2, 2)
plt.hist(errors, bins=50, edgecolor='black')
plt.title('Distribution of Errors')
plt.xlabel('Error (Prediction - Actual)')
plt.ylabel('Frequency')
plt.grid(True)

plt.tight_layout()
plt.show()
