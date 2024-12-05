# Cats vs Dogs Image Classification

This project demonstrates how to build a deep learning model using TensorFlow to classify images of cats and dogs. It uses Convolutional Neural Networks (CNN) to classify images and trains the model on a dataset of cats and dogs.

## Project Structure

The project consists of the following key components:

- **Data Preprocessing**: Loading, augmenting, and normalizing the image data.
- **Model Building**: Creating a convolutional neural network for binary classification.
- **Model Training**: Training the model using the preprocessed data.
- **Model Evaluation**: Evaluating the model’s performance based on validation and test data.
- **Results Visualization**: Plotting training accuracy, validation accuracy, and loss.
  
## Setup and Installation

To get started, clone this repository and install the required dependencies.

### Dependencies

- **TensorFlow** (for deep learning model and training)
- **Matplotlib** (for plotting training and validation metrics)
- **NumPy** (for numerical operations)
- **os** (for file handling)
  
Install TensorFlow (ensure you are using version 2.x):

```bash
pip install tensorflow
```
## Dataset

This project uses a dataset of cats and dogs images, which can be downloaded from [this link](https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip). The dataset is organized into the following directories:

- `train/` – Contains subdirectories `cats/` and `dogs/` for training images.
- `validation/` – Contains subdirectories `cats/` and `dogs/` for validation images.
- `test/` – Contains test images to evaluate the model.

To begin, download and unzip the dataset:

```bash
!wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip
!unzip cats_and_dogs.zip
```
## Image Preprocessing

The images are preprocessed using the `ImageDataGenerator` to normalize pixel values (scaled to the range [0, 1]) and apply data augmentation techniques like flipping, shifting, and zooming.

```python
train_image_generator = ImageDataGenerator(rescale=1/255, fill_mode="nearest", 
                                           horizontal_flip=True, vertical_flip=True, 
                                           width_shift_range=0.3, height_shift_range=0.3,
                                           zoom_range=0.25, shear_range=0.25)
```
## Model Architecture

The model is built using a Convolutional Neural Network (CNN) architecture. It consists of the following layers:

- **Conv2D Layer**: To extract features from images.
- **MaxPooling2D Layer**: To downsample and reduce spatial dimensions.
- **Flatten Layer**: To flatten the 3D feature maps into 1D.
- **Dense Layer**: Fully connected layer to classify the images into cat or dog.

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(2))  # 2 classes: cat and dog
model.compile(optimizer='adam', metrics=['accuracy'], loss=SparseCategoricalCrossentropy(from_logits=True))
```
## Model Training

The model is trained using the following parameters:

- **Batch size**: 128
- **Epochs**: 20
- **Image size**: 150x150 pixels
- **Training data**: 80% of images
- **Validation data**: 20% of images

```python
history = model.fit(train_data_gen, steps_per_epoch=total_train//batch_size, epochs=epochs,
                    validation_data=val_data_gen, validation_steps=total_val//batch_size, verbose=1)
```
## Model Evaluation

After training, the model’s performance is evaluated using the validation and test datasets. Training and validation accuracy, as well as loss values, are plotted.

```python
# Plot training and validation accuracy and loss
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
## Accuracy Evaluation

After training, the model’s performance is further evaluated by comparing its predictions against a set of known answers. The challenge is passed if the model achieves a classification accuracy of at least 63%.

```python
# Calculate percentage of correctly identified images
percentage_identified = (correct / len(answers)) * 100
passed_challenge = percentage_identified >= 63
```
