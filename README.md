# SVM_ADMM-Distributed

<!-- Project README scaffold: https://github.com/MRColorR/project-template -->

[Overview](#overview) | [Getting started](#getting-started) | [Documentation](#documentation) | [Contributing](#contributing) | [Support](#support) | [License](#license)

## Overview

Solve a SVM fitting problem in MATLAB using a consensus distributed ADMM approach and CVX.

The main focus of this project is to train a linear SVM and show data separated by the best hyperplane in a figure. Then use some of the data not used in the training to show in another figure the performance of the SVM trained in prediction against the real labels. The important thing is that the first SVM is written from scratch resolving the SVM fitting problem starting from the theory so it doesn't use fitcsvm but it solve a distributed convex optimization problem using distributed ADMM with split by data approach and CVX (to research the best parameters for the SVM in order to train it).

- [x] Leave a star ⭐ if you like this project 🙂 thank you.

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

Ko-fi supports Mattia / MrColorRain's work across projects.

Support ongoing development, testing and documentation through [Ko-fi](https://ko-fi.com/mrcolorrain). One-time contributions and monthly memberships are welcome, and you can pledge above a tier's minimum.

Your contribution helps me keep building, testing and sharing useful work. Explore the membership benefits on Ko-fi and choose the way you would like to support. Technical decisions remain independent.

See the [supporters and sponsors directory](https://github.com/MRColorR/MRColorR/blob/main/SPONSORS.md) for acknowledgments.

## License

A project license has not been declared in this repository. Contact the maintainer about reuse permissions.
