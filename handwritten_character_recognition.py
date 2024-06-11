# Importing the required libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Loading the dataset
data_path = r"E:\A_Z Handwritten Data.csv"
data = pd.read_csv(data_path).astype('float32')

# Preparing the features and labels
X = data.drop('0', axis=1)
y = data['0']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshaping the data for visualization
X_train_reshaped = np.reshape(X_train.values, (X_train.shape[0], 28, 28))
X_test_reshaped = np.reshape(X_test.values, (X_test.shape[0], 28, 28))

print("Training data shape:", X_train_reshaped.shape)
print("Testing data shape:", X_test_reshaped.shape)

# Mapping numerical labels to alphabets
alphabet_dict = {i: chr(65 + i) for i in range(26)}

# Counting the occurrences of each letter in the dataset
y_train_counts = y_train.value_counts().sort_index()

# Plotting the distribution of the letters
plt.figure(figsize=(10, 8))
plt.barh(list(alphabet_dict.values()), y_train_counts)
plt.xlabel("Number of Samples")
plt.ylabel("Alphabets")
plt.title("Distribution of Alphabets in the Dataset")
plt.grid(True)
plt.show()

# Shuffling a subset of training data for visualization
shuffled_data = shuffle(X_train_reshaped[:100])

# Displaying some shuffled training images
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(shuffled_data[i], cmap='gray')
    ax.axis('off')
plt.show()

# Reshaping the data for the model
X_train = X_train_reshaped.reshape(X_train_reshaped.shape[0], 28, 28, 1)
X_test = X_test_reshaped.reshape(X_test_reshaped.shape[0], 28, 28, 1)

print("Reshaped training data:", X_train.shape)
print("Reshaped testing data:", X_test.shape)

# Converting labels to one-hot encoding
y_train_ohe = to_categorical(y_train, num_classes=26)
y_test_ohe = to_categorical(y_test, num_classes=26)

print("One-hot encoded training labels shape:", y_train_ohe.shape)
print("One-hot encoded testing labels shape:", y_test_ohe.shape)

# Building the CNN model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))

# Compiling the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Defining callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Training the model
history = model.fit(X_train, y_train_ohe, epochs=10, validation_data=(X_test, y_test_ohe), callbacks=[reduce_lr, early_stop])

# Saving the model
model.save('handwritten_character_recognition_model.h5')

# Displaying model summary
model.summary()

# Plotting training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(history.history['accuracy'], label='Train Accuracy')
ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
ax1.set_title('Accuracy')
ax1.legend()

ax2.plot(history.history['loss'], label='Train Loss')
ax2.plot(history.history['val_loss'], label='Val Loss')
ax2.set_title('Loss')
ax2.legend()

plt.show()

# Making predictions on test data
predictions = model.predict(X_test[:9])

# Displaying some predictions
fig, axes = plt.subplots(3, 3, figsize=(8, 9))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(np.reshape(X_test[i], (28, 28)), cmap='gray')
    predicted_label = alphabet_dict[np.argmax(predictions[i])]
    ax.set_title(f"Prediction: {predicted_label}")
    ax.axis('off')
plt.show()
