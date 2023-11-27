from sklearn.svm import SVR 
from sklearn import metrics
import numpy as np
import scipy.stats as stats
import statistics


# Function to train a SVM RBF regressor and output its Tau and MAE.
def SVMRBFTrain(**kwargs):
    # Unpack kwargs
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    X_test = kwargs['X_test']
    y_test = kwargs['y_test']

    # Local variables for logging results
    epoch_log = [[],[]]

    # Calculate result 5 times for significance
    for i in range(5):
        svr = SVR(kernel='rbf') 
        svr.fit(X_train, np.ravel(y_train))     
        y_pred = svr.predict(X_test)  

        epoch_MAE = metrics.mean_absolute_error(y_test,y_pred)
        epoch_Tau = stats.kendalltau(y_test, y_pred)[0]

        epoch_log[0].append(epoch_MAE)
        epoch_log[1].append(epoch_Tau)
    
    MAE_average = statistics.mean(epoch_log[0])
    Tau_average = statistics.mean(epoch_log[1])

    MAE_deviation = statistics.pstdev(epoch_log[0]) 
    Tau_deviation = statistics.pstdev(epoch_log[1]) 

    return [(MAE_average,MAE_deviation),(Tau_average,Tau_deviation)]