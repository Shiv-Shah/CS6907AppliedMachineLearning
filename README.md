# Hybrid Models and Breast Cancer Masking

This repository contains code for _Hybrid Models and Breast Cancer Masking_. Source code for training models to estimate the mammographic masking levels are made available here.

Right now this resposity contains only the code for the 6 classic machine learning models we are applying to the CSAW-M dataset.

Completed Models: KNN, RBF SVM, Linear Regression, Random Forrest, Gradient Boost, and Ada Boost

To Be Completed Models: EfficientNet-B4, ConvNext-T, Twins-SVT-S, PoolFormer-S36, and MedVit

---
## Installation

##### Requirements

The only packages needed are contained in the 'general_requirements.txt' file.
To install run the following:

```bash
pip3 install -r general_requirements.txt 
```

---

## Training and evaluation

To training and test algorithms, run the following while in the CS6907AppliedMachineLearning directory:
```bash
python3 main.py
```
---

## CSAW-M Dataset

- The authors of the CSAW-M dataset have requested that we not share the files without their explicit permission. In order to keep with this request, we have provided intermediary files that obscure the original information(This is only necessary our class evaluations, durring actual submissions to conferences the dataset can be requested at: https://figshare.scilifelab.se/articles/dataset/CSAW-M_An_Ordinal_Classification_Dataset_for_Benchmarking_Mammographic_Masking_of_Cancer/14687271)
- Data can befound under CSAW_M_Subset/lables/CSAW-M_train and CSAW_M_Subset/lables/CSAW-M_test.
- In the original CSAW-M dataset, the data is pre-split into test and train sets. To keep inline with their tests, we used the same test and train files.

---

### Questions or suggestions?

Please feel free to contact us in case you have any questions or suggestions!
