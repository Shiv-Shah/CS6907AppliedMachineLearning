import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from efficientnet_pytorch import EfficientNet
import torch.utils.data as data_utils
from PIL import Image
import pandas as pd
from sklearn import metrics
import scipy.stats as stats
import statistics
import numpy as np
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

from sklearn.metrics import mean_absolute_error

def ViT_L(**kwargs):
    # Input size for model
    new_width = 224  # Set the desired width
    new_height = 224  # Set the desired height

    # Unpack image loaction
    train_path = kwargs['train_path']
    test_path = kwargs['test_path']

    # Define your dataset and dataloaders
    data_transform = transforms.Compose([
        transforms.Resize((new_width, new_height)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transform)
    val_dataset = datasets.ImageFolder(root=test_path, transform=data_transform)
    
    # Limit size to speed up training 
    indices_train = torch.arange(4000)
    indices_test = torch.arange(400)

    train_dataset_sample = data_utils.Subset(train_dataset,indices_train)
    test_dataset_sample = data_utils.Subset(val_dataset,indices_test)
    train_loader = DataLoader(train_dataset_sample, batch_size=16, shuffle=True, num_workers=8)
    val_loader = DataLoader(test_dataset_sample, batch_size= 16, shuffle=False, num_workers=8)


    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)
    WeightsEnum.get_state_dict = get_state_dict

    # Instantiate the model
    print(f"Is Cuda supported: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vit_l_32(weights=models.ViT_L_32_Weights.IMAGENET1K_V1)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0000005)

    # Training loop
    num_runs = 5     # You can increase this number for better training
    num_epochs = 15  # You can increase this number for better training

    for run in range(num_runs):
        for epoch in range(num_epochs):
            # Local variables for logging results inside epochs
            epoch_log = [[],[]]
            model.train()
            running_loss = 0.0

            # Train for epoch
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

            # Test for epoch
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                all_preds = []
                all_labels = []
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    inputs, labels = inputs.to('cpu'), labels.to('cpu')
                    predicted = predicted.to('cpu')
                    [all_preds.append(i) for i in predicted]
                    [all_labels.append(i) for i in labels]

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            # Log data

            # Calc tau
            tau, p_value = stats.kendalltau(all_preds, all_labels)
            epoch_Tau = tau

            # Calc mae
            label_set = list(set(all_labels))
            all_mae = []
            for label in label_set:
                index_list = [i for i, x in enumerate(all_labels) if x == label]
                pred_list = [all_preds[i] for i in index_list]
                label_list = [all_labels[i] for i in index_list]
                mae = mean_absolute_error(pred_list, label_list)
                all_mae.append(mae)
            epoch_MAE = np.average(all_mae)

            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy * 100:.2f}%")

            epoch_log[0].append(epoch_MAE)
            epoch_log[1].append(epoch_Tau)


    MAE_average = statistics.mean(epoch_log[0])
    Tau_average = statistics.mean(epoch_log[1])

    MAE_deviation = statistics.pstdev(epoch_log[0])
    Tau_deviation = statistics.pstdev(epoch_log[1])

    # Save the trained model
    torch.save(model.state_dict(), 'convNextT.pth')
    torch.cuda.empty_cache()
    return [(MAE_average,MAE_deviation),(Tau_average,Tau_deviation)]
    