#QUESTION 2 TASK 
import tensorflow as tf
import numpy as np

# Define the 5×5 input matrix
input_matrix = np.array([[1, 2, 3, 4, 5],
                         [6, 7, 8, 9, 10],
                         [11, 12, 13, 14, 15],
                         [16, 17, 18, 19, 20],
                         [21, 22, 23, 24, 25]], dtype=np.float32)

# Reshape to fit TensorFlow's input format [batch_size, height, width, channels]
input_tensor = tf.constant(input_matrix.reshape(1, 5, 5, 1))

# Define the 3×3 kernel
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)

# Reshape kernel to fit TensorFlow's filter format [height, width, in_channels, out_channels]
kernel_tensor = tf.constant(kernel.reshape(3, 3, 1, 1))

# Function to perform convolution and print results
def perform_convolution(stride, padding):
    output = tf.nn.conv2d(input_tensor, kernel_tensor, strides=[1, stride, stride, 1], padding=padding)
    print(f"Output with stride={stride}, padding='{padding}':\n{tf.squeeze(output).numpy()}\n")

# Perform convolution with the specified parameters
print("Convolution Results:\n")
perform_convolution(1, 'VALID')
perform_convolution(1, 'SAME')
perform_convolution(2, 'VALID')
perform_convolution(2, 'SAME')

#QUESTION 3 TASK 1

import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

# Upload image file
uploaded = files.upload()
image_path = list(uploaded.keys())[0]

# Load the grayscale image
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Verify if the image was loaded correctly
if image is None:
    raise FileNotFoundError("Image not found or could not be loaded. Please check the file path.")

# Apply Sobel filter in the x-direction and y-direction
sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

# Convert the results to uint8 for display
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)

# Display the original and filtered images
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sobel_x, cmap='gray')
plt.title('Sobel-X Edge Detection')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(sobel_y, cmap='gray')
plt.title('Sobel-Y Edge Detection')
plt.axis('off')

plt.tight_layout()
plt.show()

# QUESTION 3 TASK 2 

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate a random 4x4 matrix as input image
input_image = np.random.random((1, 4, 4, 1))  # Shape (batch, height, width, channels)

# Max Pooling (2x2)
max_pooling_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
max_pooled_image = max_pooling_layer(input_image)

# Average Pooling (2x2)
average_pooling_layer = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))
average_pooled_image = average_pooling_layer(input_image)

# Print the original matrix
print("Original Input Image (4x4):")
print(input_image[0, :, :, 0])

# Print the max-pooled matrix
print("\nMax Pooled Image (2x2 Max Pooling):")
print(max_pooled_image[0, :, :, 0])

# Print the average-pooled matrix
print("\nAverage Pooled Image (2x2 Average Pooling):")
print(average_pooled_image[0, :, :, 0])

# Add text to the images to show their values
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original Image with values annotated
axes[0].imshow(input_image[0, :, :, 0], cmap='gray')
for i in range(4):
    for j in range(4):
        axes[0].text(j, i, f"{input_image[0, i, j, 0]:.2f}", ha='center', va='center', color='red', fontsize=8)
axes[0].set_title("Original Image (4x4)")
axes[0].axis('off')

# Max Pooled Image with values annotated
axes[1].imshow(max_pooled_image[0, :, :, 0], cmap='gray')
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, f"{max_pooled_image[0, i, j, 0]:.2f}", ha='center', va='center', color='red', fontsize=8)
axes[1].set_title("Max Pooled Image (2x2)")
axes[1].axis('off')

# Average Pooled Image with values annotated
axes[2].imshow(average_pooled_image[0, :, :, 0], cmap='gray')
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, f"{average_pooled_image[0, i, j, 0]:.2f}", ha='center', va='center', color='red', fontsize=8)
axes[2].set_title("Average Pooled Image (2x2)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# QUESTION 4 TASK 1


import tensorflow as tf
from tensorflow.keras import layers, models

# Define the AlexNet model
def alexnet_model():
    model = models.Sequential()
    
    # Conv2D Layer: 96 filters, kernel size = (11, 11), stride = 4, activation = ReLU
    model.add(layers.Conv2D(96, kernel_size=(11, 11), strides=4, activation='relu', input_shape=(227, 227, 3)))
    
    # MaxPooling Layer: pool size = (3, 3), stride = 2
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
    
    # Conv2D Layer: 256 filters, kernel size = (5, 5), activation = ReLU
    model.add(layers.Conv2D(256, kernel_size=(5, 5), activation='relu'))
    
    # MaxPooling Layer: pool size = (3, 3), stride = 2
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
    
    # Conv2D Layer: 384 filters, kernel size = (3, 3), activation = ReLU
    model.add(layers.Conv2D(384, kernel_size=(3, 3), activation='relu'))
    
    # Conv2D Layer: 384 filters, kernel size = (3, 3), activation = ReLU
    model.add(layers.Conv2D(384, kernel_size=(3, 3), activation='relu'))
    
    # Conv2D Layer: 256 filters, kernel size = (3, 3), activation = ReLU
    model.add(layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    
    # MaxPooling Layer: pool size = (3, 3), stride = 2
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=2))
    
    # Flatten Layer
    model.add(layers.Flatten())
    
    # Fully Connected (Dense) Layer: 4096 neurons, activation = ReLU
    model.add(layers.Dense(4096, activation='relu'))
    
    # Dropout Layer: 50%
    model.add(layers.Dropout(0.5))
    
    # Fully Connected (Dense) Layer: 4096 neurons, activation = ReLU
    model.add(layers.Dense(4096, activation='relu'))
    
    # Dropout Layer: 50%
    model.add(layers.Dropout(0.5))
    
    # Output Layer: 10 neurons, activation = Softmax
    model.add(layers.Dense(10, activation='softmax'))
    
    return model

# Instantiate the model
model = alexnet_model()

# Print the model summary
model.summary()

#QUESTION 4 TASK 2


import tensorflow as tf
from tensorflow.keras import layers, models

# Function to define a residual block
def residual_block(input_tensor, filters):
    # First Conv2D layer with ReLU activation
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)
    # Second Conv2D layer with ReLU activation
    x = layers.Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    # Skip connection: adding the input tensor to the output of the second Conv2D layer
    x = layers.add([x, input_tensor])
    return x

# Build the ResNet-like model
def resnet_model():
    inputs = layers.Input(shape=(224, 224, 3))  # Input layer
    
    # Initial Conv2D layer
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', activation='relu')(inputs)
    
    # First residual block
    x = residual_block(x, 64)
    
    # Second residual block
    x = residual_block(x, 64)
    
    # Flatten layer to flatten the output
    x = layers.Flatten()(x)
    
    # Dense layer with 128 neurons and ReLU activation
    x = layers.Dense(128, activation='relu')(x)
    
    # Output layer with 10 neurons and Softmax activation for classification
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create the model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    return model

# Instantiate the model
model = resnet_model()

# Print the model summary
model.summary()
