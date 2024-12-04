# Section 1
# Import necessary libraries
try:
  # %tensorflow_version only works in Colab.
  !pip install tf-nightly
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Print the TensorFlow version to ensure it is correctly installed
print(tf.__version__)

# Section 2
# Download the dataset files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

# Load the training and test datasets
train_data = pd.read_csv(train_file_path, sep='\t', header=None, names=['label', 'message'])
test_data = pd.read_csv(test_file_path, sep='\t', header=None, names=['label', 'message'])

# Verify the first few rows of the data to ensure correct loading
print("First rows of the training data:\n", train_data.head())

# Section 3
# Convert the "ham" and "spam" labels to numerical values (0 and 1)
train_data['label'] = train_data['label'].map({'ham': 0, 'spam': 1})
test_data['label'] = test_data['label'].map({'ham': 0, 'spam': 1})

# Use TfidfVectorizer to convert text messages into numerical vectors
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1, 2))

# Fit the vectorizer using the training data only
X_train = vectorizer.fit_transform(train_data['message'])
X_test = vectorizer.transform(test_data['message'])

# Get the labels (Y)
y_train = train_data['label']
y_test = test_data['label']

print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# Section 4
from tensorflow import keras

# Create the neural network model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Input layer with 64 neurons
    keras.layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    keras.layers.Dense(1, activation='sigmoid')  # Output layer with sigmoid activation (probability between 0 and 1)
])

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model on the training data
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")

# Section 5
# Function to predict messages based on the trained model
# (should return a list containing prediction and label, e.g., [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    # Convert the input text to TF-IDF features
    message_tfidf = vectorizer.transform([pred_text])

    # Make the prediction using the model
    prediction = model.predict(message_tfidf)

    # Extract the scalar value of the prediction (probability)
    prediction_value = prediction[0][0]  # Extracts the scalar value (probability)

    # Return the probability and the label ('ham' or 'spam')
    label = 'spam' if prediction_value >= 0.5 else 'ham'
    return [float(prediction_value), label]

# Example of usage
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)  # This will print the probability and label (ham or spam)

# Section 6
# Test predictions with multiple messages
test_messages = [
    "Congrats! You've won a prize! Call now!",  # spam
    "Hey, how are you doing? Are we meeting tomorrow?",  # ham
    "URGENT: Your account has been compromised, click to secure it now!",  # spam
    "Can we reschedule? I'm not feeling well today."  # ham
]

# Display predictions for each message
for msg in test_messages:
    prediction = predict_message(msg)
    print(f"Message: {msg}\nPrediction: {prediction}\n")

# Section 7
# Run this cell to test your function and model. Do not modify contents.
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  # Compare the predictions to the expected answers
  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

  if passed:
    print("You passed the challenge. Great job!")
  else:
    print("You haven't passed yet. Keep trying.")

test_predictions()
