from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import scipy.stats as stats
import statistics
import numpy as np

# Function to train Random Forest regressor and output its Tau and MAE.
def RadomForestTrain(**kwargs):
    # Unpack kwargs
    X_train = kwargs['X_train']
    y_train = kwargs['y_train']
    X_test = kwargs['X_test']
    y_test = kwargs['y_test']

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
        regr.fit(X_train, np.ravel(y_train))     
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

    #print(f'Random Forrest: MAE: {best_MAE} +/- {MAE_deviation}  Kendall\'s Tau: {best_tau} +/- {tau_deviation}')
    return [(best_MAE,MAE_deviation),(best_tau,tau_deviation)]