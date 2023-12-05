import os
import pandas as pd
from PIL import Image

def resize_and_break(input_folder, output_folder, new_width, new_height):
    # Create output folder and subfolders(classes) if they doesn't exist
    outputs = ['class_1','class_2','class_3','class_4','class_5','class_6','class_7','class_8']
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    for i in outputs:
        folder_path = os.path.join(output_folder, i)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # Loop through all files in the input folder
    if input_folder == 'images/preprocessed/train':
        csv_train = os.path.join(input_folder, "CSAW-M_train.csv")
        df_label = pd.read_csv(csv_train,sep = ';')
    else:
        csv_test = os.path.join(input_folder, "CSAW-M_test.csv")
        df_label = pd.read_csv(csv_test,sep = ';')


    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # Add more extensions if needed
            # Grab label for file
            label = df_label.loc[df_label['Filename'] == filename, 'Label'].values[0]

            # Assign output folder based on label
            folder_path = os.path.join(os.path.join(output_folder, 'class_'+str(label)))

            # Construct full file paths
            input_path = os.path.join(input_folder, filename)

            # Open the image
            image = Image.open(input_path)

            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.LANCZOS)

            # Get the file name (without extension) from the input path
            file_name = os.path.splitext(filename)[0]

            # Save the resized image to the correct label folder
            output_path = os.path.join(folder_path, f"{file_name}_resized.jpg")
            resized_image.save(output_path)