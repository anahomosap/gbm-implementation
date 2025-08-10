# gbm-implementation

## Gradient Boosting Machine From Scratch

Utilizing UCI ML Repo's dataset on Credit Card recall, I implemented a GBM from scratch to be able to predict whether or not a client's credit card line will be defaulted. 

At the end of the notebook, there is a comparison of performance metrics between my implementation and scikit-learn's GradientBoostingClassifier using the same hyperparameters (# of learners/boosting stages, learning rate, tree depth/max tree depth). The main purpose of this project was to better understand the math and algorithms used to build Regression Trees and Gradient Boosting Machines.

#### Understanding GBMS:

- The GBM is built by sequentially creating learners, in this particuar case regression trees, to more accurately make predictions, particularly for the test data

- Regression trees: Each regression tree in the GBM was built using the CART algorithm (specifically for regression trees, I used residual reduction, so minimizing the residual sum of squares). Because the classes were imbalanced, there is a penalty for incorrectly predicting target value of 1 (incur penalty 4 times more than if incorrectly predicting a class 0 customer) -- the GBM was able to train faster (needed 100 learners vs 300 learners to obtain the same performance metrics). 


- The GBM builds each of these regression tree sequentially, and with some learning rate creates the final predicted probabilities that each particular client will have their credit card line defaulted 

- The final results yield similar performance metrics between the hard coded and built-in GradientBoostingClassifier with the same hyperparameters

