from PIL import Image
import os

def resize_and_save(input_folder, output_folder, new_width, new_height):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # Add more extensions if needed
            # Construct full file paths
            input_path = os.path.join(input_folder, filename)

            # Open the image
            image = Image.open(input_path)

            # Resize the image
            resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

            # Get the file name (without extension) from the input path
            file_name = os.path.splitext(filename)[0]

            # Save the resized image to the output folder
            output_path = os.path.join(output_folder, f"{file_name}_resized.jpg")
            resized_image.save(output_path)