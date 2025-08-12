## Gradient Boosting Machine From Scratch

Utilizing UCI ML Repo's dataset on Credit Card recall, implemented a GBM from scratch to be able to predict whether or not a client's credit card line will be defaulted. 

At the end of the notebook, there is a comparison of performance metrics between my implementation and scikit-learn's GradientBoostingClassifier using the same hyperparameters (# of learners/boosting stages, learning rate, tree depth/max tree depth). The main purpose of this project was to better understand the math and algorithms used to build Regression Trees and Gradient Boosting Machines.

#### Understanding GBMS:

- The GBM is built by sequentially creating weak learners based on the results of each previous learner, in this particuar case regression trees, to more accurately make predictions, particularly for the test data

- Regression trees: Each regression tree in the GBM was built using the CART algorithm (specifically for regression trees, I used residual reduction, so minimizing the residual sum of squares). Because the classes were imbalanced, there is a penalty for incorrectly predicting target value of 1 (incur penalty 4 times more than if incorrectly predicting a class 0 customer) -- the GBM was able to train faster (needed 100 learners vs 300 learners to obtain similar performance metrics).

- For each residual predicted at a leaf node, the algorithm then finds the values to minimize each loss function, which is approximately the sum of the residuals in that particular leaf, divided by the sum of each product of the predicted probability times 1 - the predicted probability

- This result is the log odds prediction for each client (example below for client m) in that leaf at that particular number learner, which is then mulitplied by the learning rate ($\alpha$) and added to the initial prediction ($F_{m, 0}$). The final prediction is the log odds prediction for an individual client ($L_{m}$), that can then be converted using the inverse logit function to get the predicted probability that a particular client's credit card line will be defaulted

- $L_{m} = F_{m, 0} + \alpha * F_{m, 1} + \alpha * F_{m, 2} + ... + \alpha * F_{m, n}$


