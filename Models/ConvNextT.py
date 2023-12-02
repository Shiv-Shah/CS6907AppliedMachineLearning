from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.convnext import ConvNeXtTiny
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import numpy as np


def ConvNextT():
    datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
    
    train_generator = datagen.flow_from_directory(
    directory = 'images/preprocessed/train_cropped/',
    target_size=(225, 225),
    batch_size=32,
    class_mode = None

)
    
    model = ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=(255, 255, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=10)
    
    # Assuming you have a test generator as well
    test_generator = datagen.flow_from_directory(
    'images/preprocessed/test_cropped/',
    target_size=(225, 225),
    batch_size=32,
    class_mode = None
    )
    
    # Make predictions on the test data
    predictions = model.predict(test_generator)

    # Print the first few predictions
    print("Predictions:")
    print(predictions[:5])

    # Get class indices from the generator
    class_indices = test_generator.class_indices

    # Convert predictions to class labels
    predicted_classes = [list(class_indices.keys())[i.argmax()] for i in predictions]

    # Get true labels from the generator
    true_labels = test_generator.classes

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

def ConvNextTV2():
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
    

   
    model = ConvNeXtTiny(weights='imagenet', include_top=False, input_shape=(255, 255, 3))
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



ConvNextT()