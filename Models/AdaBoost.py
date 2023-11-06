from sklearn.ensemble import AdaBoostRegressor
from sklearn import metrics
import scipy.stats as stats
import statistics
import numpy as np

# Function to train Ada Boost regressor and output its Tau and MAE.
def AdaBoostTrain(**kwargs):
    # Unpack kwargs
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    X_test = kwargs['X_test']
    y_test = kwargs['y_test']

    # Local variables for logging results
    epoch_log = [[],[]]

    # Calculate result 5 times for significance
    for i in range(5):
        regr = AdaBoostRegressor(random_state=0)
        regr.fit(X_train, np.ravel(y_train))     
        y_pred = regr.predict(X_test)  

        epoch_MAE = metrics.mean_absolute_error(y_test,y_pred)
        epoch_Tau = stats.kendalltau(y_test, y_pred)[0]

        epoch_log[0].append(epoch_MAE)
        epoch_log[1].append(epoch_Tau)
    
    MAE_average = statistics.mean(epoch_log[0])
    Tau_average = statistics.mean(epoch_log[1])

    MAE_deviation = statistics.pstdev(epoch_log[0]) 
    Tau_deviation = statistics.pstdev(epoch_log[1]) 

    return [(MAE_average,MAE_deviation),(Tau_average,Tau_deviation)]