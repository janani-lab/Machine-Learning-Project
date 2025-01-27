'''
Project Title: HandWritten Digit Recognition Using MNIST Dataset
Team Members: Janani Priya R
Date:19/11/2024
'''

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers, models
from keras.datasets import mnist
from keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Check the shape of the training and testing data
print("Training images shape:", train_images.shape)
print("Training labels shape:", train_labels.shape)
print("Testing images shape:", test_images.shape)
print("Testing labels shape:", test_labels.shape)

# Display unique classes in labels and their counts
unique, counts = np.unique(train_labels, return_counts=True)
print("Class distribution in training data:", dict(zip(unique, counts)))

# Plotting a grid of sample images with their labels
fig, axes = plt.subplots(3, 5, figsize=(10, 6))
axes = axes.ravel()

for i in range(15):
    index = np.random.randint(0, len(train_images))
    axes[i].imshow(train_images[index], cmap='gray')
    axes[i].set_title(f"Label: {train_labels[index]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Flatten the images and plot the distribution of pixel intensities
pixel_values = train_images.ravel()
plt.hist(pixel_values, bins=256, color='blue', alpha=0.7)
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Reshaping and normalizing the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# One-hot encoding the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Building the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Summarizing the model
model.summary()

# Compiling the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=5, batch_size=64)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Evaluate the model on the test set
train_loss, train_acc = model.evaluate(train_images, train_labels)
print("Train Accuracy: ", train_acc)
print("Train Loss: ", train_loss)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test Accuracy: ", test_acc)
print("Test Loss: ", test_loss)

# Generate predictions on the test set
predictions = model.predict(test_images)

# Convert predictions and true labels back to class labels
predicted_labels = np.argmax(predictions, axis=1)  # Predictions are one-hot encoded
true_labels = np.argmax(test_labels, axis=1)       # Test labels are one-hot encoded

# Generate the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
disp.plot(cmap=plt.cm.Blues, colorbar=True)
plt.title("Confusion Matrix")
plt.show()


# Saving the model
model.save('mnist.h5')
