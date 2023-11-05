
import pandas as pd 
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




    

#print(KNNTrain(**data_object))
#print(SVMRBFTrain(**data_object))
#print(RadomForestTrain(**data_object))
#print(GradientBoostTrain(**data_object))
print(AdaBoostTrain(**data_object))
#SVMTrain()
#RandomForrestTrain()