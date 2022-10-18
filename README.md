# SVM_ADMM-Distributed
Solve a SVM fitting problem in MATLAB using a consensus distributed ADMM approach and CVX.

The main focus of this project is to train a linear SVM and show data separated by the best hyperplane in a figure. Then use some of the data not used in the training to show in another figure the performance of the SVM trained in prediction against the real labels. The important thing is that the first SVM is written from scratch resolving the SVM fitting problem starting from the theory so it doesn't use fitcsvm but it solve a distributed convex optimization problem using distributed ADMM with split by data approach and CVX (to research the best parameters for the SVM in order to train it).

## What's inside
- Two linear SVMs one distributed, one centralized.
- Small sample dataset
- Random dataset generator
- performance and data plots  
- project document with details on the formulas and theory used
- CVX installer (sept-2022)

The first SVM is decentralized and is written from scratch starting from the fitting problem of SVM solved in a distributed form through ADMM and CVX. The second one is the classic svm made with fitcsvm for comparison.  
The code is written so that they can be easily re-trained on different datasets even with different number of features.
