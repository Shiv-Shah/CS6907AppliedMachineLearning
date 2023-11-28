import torch
#from datasets import load_dataset
#from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
# from Utility.CroppingImage import resize_and_save



# # Example usage
# input_folder_path_train = "images/preprocessed/train"
# output_folder_path_train = "images/preprocessed/train_cropped"
# input_folder_path_test = "images/preprocessed/test"
# output_folder_path_test = "images/preprocessed/test_cropped"
# new_width = 380  # Set the desired width
# new_height = 380  # Set the desired height

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
    class_mode='binary'
)
    
    model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=(380, 380, 3))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_generator, epochs=10)
    
    # Assuming you have a test generator as well
    test_generator = datagen.flow_from_directory(
    'images/preprocessed/test_cropped/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
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
    

EfficentNetB4()
