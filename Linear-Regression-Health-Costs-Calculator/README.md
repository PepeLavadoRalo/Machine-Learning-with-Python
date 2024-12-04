# Health Insurance Prediction using Deep Learning

This project demonstrates how to build a deep learning model to predict the insurance expenses of individuals based on their characteristics (e.g., age, BMI, smoking habits, etc.). It uses a dataset of health costs (`insurance.csv`) and performs data preprocessing, feature engineering, training a neural network model, and evaluating its performance.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Steps](#steps)
    - [1. Download and Load the Dataset](#1-download-and-load-the-dataset)
    - [2. Data Preprocessing](#2-data-preprocessing)
    - [3. Split the Dataset](#3-split-the-dataset)
    - [4. Model Building](#4-model-building)
    - [5. Training the Model](#5-training-the-model)
    - [6. Evaluation](#6-evaluation)
    - [7. Visualization](#7-visualization)
- [Results](#results)
- [Usage](#usage)
- [Contributors](#contributors)
- [License](#license)

---

## Project Overview

The objective of this project is to predict medical insurance expenses using a variety of input features such as:

- Age
- BMI (Body Mass Index)
- Children (Number of children/dependents)
- Sex
- Smoker status
- Region

We use a neural network model built with Keras, a high-level neural networks API in TensorFlow. This model is trained on a dataset containing personal details and insurance charges of individuals. The trained model can predict the insurance expenses based on the input features.

---

## Dataset

The dataset used in this project is `insurance.csv`. This file contains the following columns:

- **age**: Age of the individual.
- **sex**: Gender of the individual (male/female).
- **bmi**: Body mass index of the individual.
- **children**: Number of children/dependents.
- **smoker**: Whether the individual is a smoker (yes/no).
- **region**: Geographical region of the individual (northeast, northwest, southeast, southwest).
- **expenses**: The medical expenses of the individual (target variable).

You can download the dataset from the following link:
[Download insurance dataset](https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv)

---

## Prerequisites

To run this project, you need the following dependencies:

- **Python 3.x**
- **pandas** - For data manipulation and analysis.
- **scikit-learn** - For machine learning utilities such as splitting the dataset and feature scaling.
- **tensorflow** - For building and training the deep learning model.
- **matplotlib** - For plotting the graphs.
  
You can install these dependencies using `pip`:

```bash
pip install pandas scikit-learn tensorflow matplotlib
```
## Installation
Clone the repository:
```bash
git clone https://github.com/yourusername/health-insurance-prediction.git
cd health-insurance-prediction
```
Install the required Python libraries:
```bash
pip install -r requirements.txt
```
## Steps
### 1. Download and Load the Dataset
The dataset insurance.csv is downloaded and loaded into a pandas DataFrame. This is the initial step in the process:

```python
dataset = pd.read_csv('insurance.csv')
```
### 2. Data Preprocessing
We perform the following preprocessing steps on the dataset:

Encoding categorical variables (sex, smoker, region) into numerical values. For instance:
sex (male = 0, female = 1)
smoker (yes = 1, no = 0)
region is one-hot encoded to represent geographical regions as separate binary columns.
This allows the neural network to understand these features as numerical inputs.

```python
dataset['sex'] = dataset['sex'].map({'male': 0, 'female': 1})
dataset['smoker'] = dataset['smoker'].map({'yes': 1, 'no': 0})
dataset = pd.get_dummies(dataset, columns=['region'], drop_first=True)
```

### 3. Split the Dataset
We split the dataset into training and test datasets (80% for training and 20% for testing) using the train_test_split function from scikit-learn:

```python
train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    dataset, labels, test_size=0.2, random_state=42
)
```

### 4. Model Building
A neural network model is created using Keras. The model consists of:

An input layer with 128 units (size of the input dataset).
A hidden layer with 128 units and ReLU activation.
An output layer with 1 unit (predicting a single continuous value).
```python
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[train_dataset_scaled.shape[1]]),
    layers.Dense(128, activation='relu'),
    layers.Dense(1)
])
```

### 5. Training the Model
The model is compiled using the Adam optimizer with a learning rate of 0.001. The loss function used is Mean Squared Error (MSE), which is typical for regression problems.

```python
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_error',
              metrics=['mae', 'mse'])
```
To avoid overfitting, we use EarlyStopping to monitor the validation loss and stop training if it does not improve for 50 epochs.

```python
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)
```
### 6. Evaluation
After training the model, we evaluate its performance using the test dataset. We calculate the Mean Absolute Error (MAE) and Mean Squared Error (MSE) for the test set.

```python
loss, mae, mse = model.evaluate(test_dataset_scaled, test_labels, verbose=2)
```
### 7. Visualization
Finally, we visualize the performance of the model by comparing the predicted values against the actual values and plotting the distribution of errors (residuals):

```python
# Plot predictions vs real values
plt.scatter(test_labels, predictions)
plt.plot([test_labels.min(), test_labels.max()], [test_labels.min(), test_labels.max()], color='red', lw=2)  # Identity line

# Plot distribution of errors
errors = predictions.flatten() - test_labels
plt.hist(errors, bins=50, edgecolor='black')
```

### Results
After training and evaluating the model, you can expect results like:

* Mean Absolute Error (MAE) *: Indicates the average absolute difference between predicted and actual values.
* Mean Squared Error (MSE)*: Measures the average squared difference between predicted and actual values.
The model should give you insights into how well it predicts the insurance expenses.

### Usage
Once the model is trained, you can use it to predict insurance expenses for new data points. Ensure that you preprocess the new data similarly to the training data before making predictions.

For example:

```python
new_data = [[45, 23, 2, 0, 1, 0, 1]]  # Example input data
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
print(f"Predicted insurance expense: {prediction}")
```
