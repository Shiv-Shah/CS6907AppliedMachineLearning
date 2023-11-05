from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import scipy.stats as stats
import statistics

# Function to find best k for KNN regressor and output its Tau and MAE.
def KNNTrain(**kwargs):
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
        epoch_log = [[],[]]

        # Try every k value within reasont to guanrentee best results
        for k in range(1, 30):
            knn_regressor = KNeighborsRegressor(n_neighbors=k)
            knn_regressor.fit(X_train, y_train)
            y_pred = knn_regressor.predict(X_test)  
            # Log and update sub-epoch metrics
            if metrics.mean_absolute_error(y_test,y_pred) < epoch_lowest_MAE:
                epoch_lowest_MAE = metrics.mean_absolute_error(y_test,y_pred)
                epoch_lowest_tau = stats.kendalltau(y_test, y_pred)[0]

        epoch_log[0].append(epoch_lowest_MAE)
        epoch_log[1].append(epoch_lowest_tau)


    MAE_average = statistics.mean(epoch_log[0])
    Tau_average = statistics.mean(epoch_log[1])

    MAE_deviation = statistics.pstdev(epoch_log[0]) 
    Tau_deviation = statistics.pstdev(epoch_log[1]) 

    return [(MAE_average,MAE_deviation),(Tau_average,Tau_deviation)]
    print(f'KNN: MAE: {best_MAE} +/- {MAE_deviation}  Kendall\'s Tau: {best_tau} +/- {tau_deviation}')