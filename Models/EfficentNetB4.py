#import torch
#from datasets import load_dataset
#from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow as tf
#from sklearn.model_selection import train_test_split
import pandas as pd
#from Utility.CroppingImage import resize_and_save

import numpy as np

from PIL import Image
import os

# def resize_and_save(input_folder, output_folder, new_width, new_height):
#     # Create output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
        
#     # Loop through all files in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # Add more extensions if needed
#             # Construct full file paths
#             input_path = os.path.join(input_folder, filename)

#             # Open the image
#             image = Image.open(input_path)

#             # Resize the image
#             resized_image = image.resize((new_width, new_height), Image.LANCZOS)

#             # Get the file name (without extension) from the input path
#             file_name = os.path.splitext(filename)[0]

#             # Save the resized image to the output folder
#             output_path = os.path.join(output_folder, f"{file_name}_resized.jpg")
            
            
#             resized_image.save(output_path)


# # Example usage

# input_folder_path_train = "images/preprocessed/train"
# output_folder_path_train = "images/preprocessed/train_cropped"
# input_folder_path_test = "images/preprocessed/test"
# output_folder_path_test = "images/preprocessed/test_cropped"
# new_width = 255  # Set the desired width
# new_height = 255  # Set the desired height

# resize_and_save(input_folder_path_train, output_folder_path_train, new_width, new_height)
# resize_and_save(input_folder_path_test, output_folder_path_test, new_width, new_height)

def EfficentNetB4():

    datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
    
    train_generator = datagen.flow_from_directory(
    directory = 'images/preprocessed/train_cropped/',
    target_size=(380, 380),
    batch_size=32,
    class_mode = None

)
    train_csv = pd.read_csv("C:/Users/mathw/OneDrive/Desktop/GWU/CS6907AppliedMachineLearning/labels/labels/CSAW-M_train.csv")
    test_csv = pd.read_csv("C:/Users/mathw/OneDrive/Desktop/GWU/CS6907AppliedMachineLearning/labels/labels/CSAW-M_test.csv")
    train_dataset = tf.data.Dataset.from_tensor_slices(train_generator,train_csv)
    
    model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(255, 255, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_dataset, epochs=10)
    
    # Assuming you have a test generator as well
    test_generator = datagen.flow_from_directory(
    'images/preprocessed/test_cropped/',
    target_size=(224, 224),
    batch_size=32,
    class_mode = None
    )

    test_dataset = tf.data.Dataset.from_tensor_slices(test_generator,test_csv)
    
    # Make predictions on the test data
    predictions = model.predict(test_dataset)

    # Print the first few predictions
    print("Predictions:")
    print(predictions[:5])

    # Get class indices from the generator
    class_indices = test_dataset.class_indices

    # Convert predictions to class labels
    predicted_classes = [list(class_indices.keys())[i.argmax()] for i in predictions]

    # Get true labels from the generator
    true_labels = test_dataset.classes

    # Print the first few true labels and predicted labels
    print("\nTrue Labels:")
    print(true_labels[:5])
    print("\nPredicted Labels:")
    print(predicted_classes[:5])
    
    # Evaluate the model using the test generator
    evaluation_result = model.evaluate(test_generator)

    print("Test Loss:", evaluation_result[0])
    print("Test Accuracy:", evaluation_result[1])
    


def preProcessData(input_directory):
    preprocessed_images = []
    # Loop through all files in the input folder
    for filename in os.listdir(input_directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # Add more extensions if needed
            # Construct full file paths
            input_path = os.path.join(input_directory, filename)
            img = image.load_img(input_path, target_size=(380, 380))
            img_array = image.img_to_array(img)
            img_array = preprocess_input(img_array[None, 32])

            preprocessed_images.append(img_array)
    return np.vstack(preprocessed_images)

def EfficentNetB4V2():
    input_folder_path_train = "images/preprocessed/train_cropped"
    # output_folder_path_train = "images/preprocessed/train_cropped"
    input_folder_path_test = "images/preprocessed/test_cropped"
    # output_folder_path_test = "images/preprocessed/test_cropped"

    X = preProcessData(input_folder_path_train)
    Y = preProcessData(input_folder_path_test)
    



    datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
    

   
    model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(255, 255, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, epochs=10)
    
    # Assuming you have a test generator as well
 
    
    # Make predictions on the test data
    predictions = model.predict(Y)

    # Print the first few predictions
    print("Predictions:")
    print(predictions[:5])

    # # Get class indices from the generator
    # class_indices = test_generator.class_indices

    # # Convert predictions to class labels
    # predicted_classes = [list(class_indices.keys())[i.argmax()] for i in predictions]

    # # Get true labels from the generator
    # true_labels = test_generator.classes

    # Print the first few true labels and predicted labels
    # print("\nTrue Labels:")
    # print(true_labels[:5])
    # print("\nPredicted Labels:")
    # print(predicted_classes[:5])
    
    # Evaluate the model using the test generator
    evaluation_result = model.evaluate(Y)

    print("Test Loss:", evaluation_result[0])
    print("Test Accuracy:", evaluation_result[1])



EfficentNetB4()
