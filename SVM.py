import pandas as pd

from sklearn.model_selection import train_test_split



#usecols =["Filename", "Label", "Expert_1", "Expert_2","Expert_3","Expert_4","Expert_5",
#"If_cancer","If_interval_cancer","If_large_invasive_cancer","If_composite","Dicom_image_laterality","Dicom_window_center","Dicom_window_width","Dicom_photometric_interpretation","Libra_percent_density","Libra_dense_area","Libra_breast_area"]

#usecols = ['Libra_breast_area'; 'Label'; 'Expert_2'; 'Dicom_window_width'; 'Expert_4'; 'If_interval_cancer'; 'If_large_invasive_cancer'; 
#'Dicom_window_center'; 'Libra_percent_density'; 'Expert_3'; 'Expert_1'; 'Expert_5'; 'Filename'; 'If_cancer'; 
#'Dicom_photometric_interpretation'; 'Dicom_image_laterality'; 'If_composite'; 'Libra_dense_area']



testCsvFile = pd.read_csv('/home/accidentalgenius/Desktop/14687271/CSAW-M-main/labels/CSAW-M_test.csv', sep=';' )
trainCsvFile = pd.read_csv('/home/accidentalgenius/Desktop/14687271/CSAW-M-main/labels/CSAW-M_train.csv', sep=';' )
#print(trainCsvFile)

#trainCsvFile['Filename'] = pd.to_numeric(trainCsvFile['Filename'])



train, val = train_test_split(trainCsvFile,test_size=0.2)
trainy=train["If_cancer"]
del train["If_cancer"]
valy=val["If_cancer"]
del val["If_cancer"]
testy=trainCsvFile["If_cancer"]
del trainCsvFile["If_cancer"]




print (testCsvFile.dtypes)

from sklearn import svm
model = svm.SVC()
model.fit(train,trainy)




# print(val)


