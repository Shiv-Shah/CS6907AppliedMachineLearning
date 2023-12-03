import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from efficientnet_pytorch import EfficientNet
import os 
from PIL import Image
import pandas as pd

input_folder_path_train = "images/preprocessed/train"
output_folder_path_train = "images/preprocessed/train_cropped"
input_folder_path_test = "images/preprocessed/test"
output_folder_path_test = "images/preprocessed/test_cropped"
new_width = 255  # Set the desired width
new_height = 255  # Set the desired height

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

# Uncomment to resize and label data
resize_and_break(input_folder_path_train, output_folder_path_train, new_width, new_height)
resize_and_break(input_folder_path_test, output_folder_path_test, new_width, new_height)






# Define your dataset and dataloaders
data_transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
])

# Replace 'your_dataset_directory' with the path to your dataset
train_dataset = datasets.ImageFolder(root='images/preprocessed/train_cropped', transform=data_transform)
val_dataset = datasets.ImageFolder(root='images/preprocessed/test_cropped', transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Instantiate the model
model = models.efficientnet_b4(False)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # You can increase this number for better training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'efficientnetb4_model.pth')
