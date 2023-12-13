
import pandas as pd 
from Utility.ProgressBar import progressBar
from Utility.ImageManipulation import resize_and_break
from Models.KNN import KNNTrain
from Models.RBFSVR import SVMRBFTrain
from Models.LinearRegression import LinearTrain
from Models.RandomForest import RadomForestTrain
from Models.GradientBoost import GradientBoostTrain
from Models.AdaBoost import AdaBoostTrain
from tabulate import tabulate
from Models.EfficentNetB4 import EfficentNetB4
from Models.ConvNextT import ConvNextT
from Models.ViT_L import ViT_L
import torch.utils.data as data_utils
import torch
from torchvision import transforms, datasets, models
import os
from PIL import Image

#Functioon to Correctly resize the images
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
    if input_folder == '/content/drive/MyDrive/images/preprocessed/train':
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


def classModels():

    train_path = '14687271/labels/CSAW-M_train.csv'
    test_path = '14687271/labels/CSAW-M_test.csv'

    # Load Training Data
    df = pd.read_csv(train_path, sep=';')
    df.dropna()
    X_train = df.loc[:, ['Libra_percent_density','Libra_dense_area','Libra_breast_area']]
    y_train = df.loc[:, ['Label']]

    # Load Testing Data
    df = pd.read_csv(test_path, sep=';')
    df.dropna()
    X_test = df.loc[:, ['Libra_percent_density','Libra_dense_area','Libra_breast_area']]
    y_test = df.loc[:, ['Label']]

    # Pack Test and Train Data into data object
    data_object = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}


    # Run models
    results = []
    progressBar(0,6, ' KNN Training')
    KNN_result = KNNTrain(**data_object)
    results.append(['KNN', KNN_result[0], KNN_result[1]])

    progressBar(1,6, ' SVM RBF Training')
    SVMRBF_result = SVMRBFTrain(**data_object)
    results.append(['SVM RBF', SVMRBF_result[0], SVMRBF_result[1]])

    progressBar(2,6, ' Linear Regression Training')
    Linear_result = LinearTrain(**data_object)
    results.append(['Linear Regression', Linear_result[0], Linear_result[1]])

    progressBar(3,6, ' Random Forest Training')
    RadomForest_result = RadomForestTrain(**data_object)
    results.append(['Radom Forest', RadomForest_result[0], RadomForest_result[1]])

    progressBar(4,6, ' Gradient Boost Training')
    GradientBoost_result = GradientBoostTrain(**data_object)
    results.append(['Gradient Boost', GradientBoost_result[0], GradientBoost_result[1]])

    progressBar(5,6, ' Ada Boost Training')
    AdaBoost_result = AdaBoostTrain(**data_object)
    results.append(['Ada Boost', AdaBoost_result[0], AdaBoost_result[1]])

    progressBar(6,6)

    # Format Results 
    for i in results:
        base = str('%.5f'%i[1][0])
        deviation = str(i[1][1])
        i[1] = base + ' +/- ' + deviation

        base = str('%.5f'%i[2][0])
        deviation = str(i[2][1])
        i[2] = base + ' +/- ' + deviation

    # Print Results in a table
    print('\n')
    print(tabulate(results, headers=['Model', 'Average Mean Average Error', 'Kendall’s Tau']))
    print('Results collected over 5 runs of each model')


def DeepLearning():
    #input_folder_path_train = "/content/drive/MyDrive/images/preprocessed/train"
    #output_folder_path_train = "/content/drive/MyDrive/images/preprocessed/train_cropped"
    #input_folder_path_test = "/content/drive/MyDrive/images/preprocessed/test"
    #output_folder_path_test = "/content/drive/MyDrive/images/preprocessed/test_cropped"
    input_folder_path_train = "images/preprocessed/train"
    output_folder_path_train = "images/preprocessed/train_cropped"
    input_folder_path_test = "images/preprocessed/test"
    output_folder_path_test = "images/preprocessed/test_cropped"    
    new_width = 255  # Set the desired width
    new_height = 255  # Set the desired height

    # Uncomment to resize and label data
    resize_and_break(input_folder_path_train, output_folder_path_train, new_width, new_height)
    resize_and_break(input_folder_path_test, output_folder_path_test, new_width, new_height)

    data_object = {'train_path':output_folder_path_train, 'test_path':output_folder_path_test}

    # 
    EfficentNetB4Train(**data_object)
    
DeepLearning()


