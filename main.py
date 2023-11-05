
import pandas as pd 
from Utility.ProgressBar import progressBar
from Models.KNN import KNNTrain
from Models.RBFSVR import SVMRBFTrain
from Models.RandomForest import RadomForestTrain
from Models.GradientBoost import GradientBoostTrain
from Models.AdaBoost import AdaBoostTrain

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
progressBar(0,5, ' KNN Training')
KNNTrain(**data_object)
progressBar(1,5, ' SVM RBF Training')
SVMRBFTrain(**data_object)
progressBar(2,5, ' Random Forest Training')
RadomForestTrain(**data_object)
progressBar(3,5, ' Gradient Boost Training')
GradientBoostTrain(**data_object)
progressBar(4,5, ' Ada Boost Training')
AdaBoostTrain(**data_object)
progressBar(5,5)
