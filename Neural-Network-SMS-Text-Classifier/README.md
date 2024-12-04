# SMS Spam Classification with Neural Networks

This project implements an SMS spam classification model using a neural network in TensorFlow/Keras. The model is trained to distinguish between "spam" and "ham" (non-spam) messages using natural language processing (NLP) techniques. We use a dataset of SMS messages, preprocess the text data, and then build a neural network model for binary classification.

## Project Structure

The project is divided into several sections, each of which handles a specific part of the process:

- **Section 1**: Import necessary libraries and install required dependencies.
- **Section 2**: Download and load the dataset.
- **Section 3**: Preprocess the data by converting labels to numeric values and converting text to numerical representations using TF-IDF.
- **Section 4**: Build and train a neural network model using Keras.
- **Section 5**: Implement a function to make predictions based on the trained model.
- **Section 6**: Test the model on various SMS examples to verify its predictions.
- **Section 7**: Automatically test the model against a set of predefined test cases.

## Requirements

- Python 3.x
- TensorFlow (tested with TensorFlow 2.x)
- pandas
- numpy
- scikit-learn
- matplotlib

### Install dependencies

You can install the required dependencies by running the following command:

```bash
pip install tensorflow pandas scikit-learn matplotlib
```
# Alternative Installation for Google Colab

Alternatively, if you are using Google Colab, the code will automatically install TensorFlow nightly builds.

## Dataset

The dataset used for training and testing consists of SMS messages labeled as either "ham" (non-spam) or "spam". The dataset is publicly available and can be downloaded from the following links:

- [Training Data](https://cdn.freecodecamp.org/project-data/sms/train-data.tsv)
- [Testing Data](https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv)

The data files are tab-separated values (TSV), with each row containing a label (either 'ham' or 'spam') and a corresponding SMS message.

## How to Use

1. **Download the data**: The `train-data.tsv` and `valid-data.tsv` files will be automatically downloaded in the script.

2. **Preprocess the data**: The text data will be preprocessed using the `TfidfVectorizer` from `scikit-learn`, which converts the text messages into numerical vectors using Term Frequency-Inverse Document Frequency (TF-IDF). The labels ('ham' and 'spam') are converted to binary numeric values, with 'ham' being represented as 0 and 'spam' as 1.

3. **Build the model**: A simple neural network is created using Keras with the following layers:
    - **Input layer**: A Dense layer with 64 neurons, using ReLU activation.
    - **Hidden layer**: A Dense layer with 32 neurons, using ReLU activation.
    - **Output layer**: A Dense layer with 1 neuron, using a sigmoid activation function to output a probability between 0 and 1.

4. **Train the model**: The model is trained on the training dataset (`train-data.tsv`) using binary cross-entropy loss and the Adam optimizer. The model's performance is evaluated on the test dataset (`valid-data.tsv`).

5. **Make predictions**: The trained model can be used to predict whether a given SMS message is "ham" or "spam" by calling the `predict_message()` function.

6. **Test the model**: The `test_predictions()` function automatically tests the model's predictions on several predefined messages.

## Usage Example

Here is an example of how you can use the `predict_message` function to classify an SMS message:

```python
# Example usage of the predict_message function
pred_text = "how are you doing today?"
prediction = predict_message(pred_text)
print(prediction)  # This will print the probability and label ('ham' or 'spam')
```
This will output a prediction for the message "how are you doing today?", indicating whether the message is classified as "ham" or "spam" along with the corresponding probability.

## Sections of the Code

### Section 1: Import necessary libraries
This section installs the required dependencies and imports the necessary libraries, including TensorFlow, pandas, numpy, scikit-learn, and matplotlib.

### Section 2: Download and load the dataset
In this section, the training and test datasets are downloaded from the provided URLs and loaded into pandas DataFrames.

### Section 3: Preprocess the data
The data is preprocessed by converting the text messages into numerical representations using the TfidfVectorizer. The labels are converted from text ('ham' and 'spam') to binary numeric values (0 and 1).

### Section 4: Build the model
Here, a simple neural network model is created using Keras, with 64 neurons in the input layer and 32 neurons in the hidden layer. The output layer has 1 neuron with a sigmoid activation function for binary classification.

### Section 5: Make predictions
This section defines the `predict_message` function, which takes a text message as input, converts it into numerical features using the TF-IDF vectorizer, and predicts whether the message is "ham" or "spam".

### Section 6: Test predictions
In this section, the model is tested on various predefined SMS messages to verify the accuracy of the predictions.

### Section 7: Automated testing
This section contains a test function that automatically checks if the model is working correctly by comparing its predictions against a set of predefined answers.

## Model Evaluation
The performance of the model is evaluated on the test dataset using accuracy as the evaluation metric.
The results are printed out after training.
```python
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")
```
The model's accuracy on the test data is shown as a percentage, which helps gauge its performance.

## Conclusion
This project demonstrates the power of neural networks for binary text classification, using a simple yet effective model. By leveraging TensorFlow and scikit-learn, we are able to preprocess the data, build a model, and make predictions on unseen text data.

Feel free to modify and extend the code for further experiments or improvements.
