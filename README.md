import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from skimage import io
from sklearn.model_selection import train_test_split
import cv2
from google.colab.patches import cv2_imshow  # Import the Colab patch for cv2.imshow
from PIL import Image # Import the Image class from PIL
# Load images from a folder
def load_images_from_folder(folder):
    images = []
    filenames = [] # Initialize a list to store filenames
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = io.imread(img_path)
            # Convert grayscale to RGB
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            images.append(img)
            filenames.append(filename) # Append the filename to the list
    return images, filenames # Return both images and filenames
# Preprocess images (resize, normalize, etc.)
def preprocess_images(images):
    resized_images = []
    for img in images:
        # Resize the image, ensuring it has 3 channels
        resized_img = tf.image.resize(img, (128, 128)).numpy()
        if resized_img.shape != (128, 128, 3):
            resized_img = np.resize(resized_img, (128, 128, 3))  # Resize if necessary
        resized_images.append(resized_img)
    normalized_images = np.array(resized_images) / 255.0  # Normalize all images
    return normalized_images  # Return the normalized images
# Load positive (crystals) and negative (no_crystals) samples
crystals_folder = "/content/drive/MyDrive/Crystals"
no_crystals_folder = "/content/drive/MyDrive/No Crystals"
positive_samples, _ = load_images_from_folder(crystals_folder) # Unpack the results, ignoring filenames for now
negative_samples, _ = load_images_from_folder(no_crystals_folder) # Unpack the results, ignoring filenames for now
# Create labels (1 for crystals, 0 for no_crystals)
labels = np.concatenate([np.ones(len(positive_samples)), np.zeros(len(negative_samples))])
# Preprocess images
preprocessed_positive = preprocess_images(positive_samples)
preprocessed_negative = preprocess_images(negative_samples)
preprocessed_images = np.concatenate([preprocessed_positive, preprocessed_negative])
# Split dataset into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(preprocessed_images, labels, test_size=0.2, random_state=42)
# Build a simple CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification (crystals or no_crystals)
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))
# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.4f}")
# Save the trained model
model.save("crystal_detection_model.h5")
print("Model saved as crystal_detection_model.h5")
# Load the saved model
loaded_model = models.load_model("crystal_detection_model.h5")
# Load an example image (e.g., "crystal_example.jpg")
example_image_path = "/content/drive/MyDrive/clean_crystal_images/processed_Yenway.tif"
example_image = io.imread(example_image_path)

# Check the dimensions of the image
print("Image dimensions:", example_image.shape)
# Preprocess the image, handling potential dimensionality issues
if example_image.ndim == 2:  # If grayscale, convert to RGB
    example_image = cv2.cvtColor(example_image, cv2.COLOR_GRAY2RGB)
elif example_image.ndim == 3 and example_image.shape[2] == 4:  # If RGBA, remove alpha channel
    example_image = example_image[:, :, :3]
preprocessed_example_image = preprocess_images([example_image]) # Now preprocess
# Predict using the loaded model
prediction = loaded_model.predict(preprocessed_example_image)
# Set a threshold for prediction (e.g., 0.5)
threshold = 0.1
# If the prediction score is above the threshold, consider it a crystal detection
if prediction[0][0] > threshold:
    print("Crystal detected!")
    # Draw a bounding box around the crystal
    (x, y, w, h) = (0, 0, example_image.shape[1], example_image.shape[0])
    cv2.rectangle(example_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2_imshow(example_image)
    cv2.waitKey(0)
else:
    print("No crystal detected.")
