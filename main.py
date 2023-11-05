
import pandas as pd 
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import scipy.stats as stats
import statistics
from Models.KNN import KNNTrain
from Models.RBFSVR import SVMRBFTrain

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

data_object = {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test}



def RandomForrestTrain():
    # Local variables for logging results
    best_per_epoch = [[],[]]
    best_MAE = 99999    # Set defualt to an artifically high number
    best_tau = 0

    # Calculate result 5 times for significance
    for i in range(5):
        # Local variables for logging results inside epochs
        epoch_lowest_MAE = 99999    # Set defualt to an artifically high number
        epoch_lowest_tau = 0

        regr = RandomForestRegressor(random_state=0)
        regr.fit(X_train, y_train)     
        y_pred = regr.predict(X_test)  

        if metrics.mean_absolute_error(y_test,y_pred) < epoch_lowest_MAE:
                epoch_lowest_MAE = metrics.mean_absolute_error(y_test,y_pred)
                epoch_lowest_tau = stats.kendalltau(y_test, y_pred)[0]

        best_per_epoch[0].append(epoch_lowest_MAE)
        best_per_epoch[1].append(epoch_lowest_tau)

        if epoch_lowest_MAE < best_MAE:
            best_MAE = epoch_lowest_MAE
            best_tau = epoch_lowest_tau
    
    MAE_deviation = statistics.pstdev(best_per_epoch[0]) 
    tau_deviation = statistics.pstdev(best_per_epoch[1]) 

    print(f'Random Forrest: MAE: {best_MAE} +/- {MAE_deviation}  Kendall\'s Tau: {best_tau} +/- {tau_deviation}')

    

print(KNNTrain(**data_object))
print(SVMRBFTrain(**data_object))
#SVMTrain()
#RandomForrestTrain()