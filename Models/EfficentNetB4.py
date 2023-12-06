import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
from efficientnet_pytorch import EfficientNet
import os
from PIL import Image
import pandas as pd

def EfficentNetB4Train(**kwargs):

    #Unpack kwargs
    train_dataset = kwargs['train_dataset']
    val_dataset = kwargs['val_dataset']

    # Create data loader objects
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    val_loader = DataLoader(val_dataset, batch_size= 16, shuffle=False, num_workers=6)


    #from torchvision.models import efficientnet_b4, EfficientNet_b4_Weights
    

    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)
    WeightsEnum.get_state_dict = get_state_dict

    #efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    #efficientnet_b0(weights="DEFAULT")

    # Instantiate the model
    print(f"Is Cuda supported: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)
    #model = models.efficientnet_b4(weights="DEFAULT")
    model = model.to(device)

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
    torch.cuda.empty_cache() 
    #torch.save(model.state_dict(), 'efficientnetb4_model.pth')

