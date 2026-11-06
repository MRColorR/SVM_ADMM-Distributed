# SVM_ADMM-Distributed

<!-- Project README scaffold: https://github.com/MRColorR/project-template -->

**Compare a distributed support vector machine with a conventional MATLAB implementation.**

[**Run the example**](#getting-started) · [Documentation](#documentation) · [Support](#support)

**Leave a star ⭐ if you like this project 🙂 Thank you!**

## Overview

Solve a SVM fitting problem in MATLAB using a consensus distributed ADMM approach and CVX.

The main focus of this project is to train a linear SVM and show data separated by the best hyperplane in a figure. Then use some of the data not used in the training to show in another figure the performance of the SVM trained in prediction against the real labels. The important thing is that the first SVM is written from scratch resolving the SVM fitting problem starting from the theory so it doesn't use fitcsvm but it solve a distributed convex optimization problem using distributed ADMM with split by data approach and CVX (to research the best parameters for the SVM in order to train it).


## Getting started

Start with [runMe.m](runMe.m), then use the project report under [Documentation](#documentation) for the model and optimization details.

## Documentation

- [Tesina_RM_Distributed_SVM.pdf](Tesina_RM_Distributed_SVM.pdf)
- [svm_admm.m](svm_admm.m)
- [newData.m](newData.m)
- [cvx-w64/cvx/LICENSE.txt](cvx-w64/cvx/LICENSE.txt)

<a name="whats-inside"></a>

### What's inside

- Two linear SVMs one distributed, one centralized.
- Small sample dataset
- Random dataset generator
- performance and data plots  
- project document with details on the formulas and theory used (ITA)
- CVX installer (sept-2022)

The first SVM is decentralized and is written from scratch starting from the fitting problem of SVM solved in a distributed form through ADMM and CVX. The second one is the classic svm made with fitcsvm for comparison.  
The code is written so that they can be easily re-trained on different datasets even with different number of features.

## Contributing

Ideas, bug reports, documentation fixes and focused changes are welcome through this repository's collaboration channels.

Please include enough context to reproduce a problem or review a change.

## Support

Help fund ongoing development, testing, and documentation with a one-time contribution or monthly membership on [Ko-fi](https://ko-fi.com/mrcolorrain). Technical decisions remain independent.

Thank you to the people who support MrColorRain’s work. See the [supporters and sponsors directory](https://github.com/MRColorR/MRColorR/blob/main/SPONSORS.md).

## License

A project license has not been declared in this repository. Contact the maintainer about reuse permissions.
