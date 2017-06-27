# README #

This is the IPython repository for the Machine Learning: Classification course of Coursera. We will use stochastic gradient ascent, ensemble methods, and Adaboost for classification problems.

### Week 1: Use product review data from Amazon.com to predict whether the sentiments about a product (from its reviews) are positive or negative, using logistic regression. ###
    * Use SFrames to do some feature engineering, and train a logistic regression model to predict the sentiment of product reviews.
    * Inspect the weights (coefficients) of a trained logistic regression model, and interpret their meanings.
    * Make a prediction (both class and probability) of sentiment for a new product review.
    * Given the logistic regression weights, predictors and ground truth labels, write a function to compute the accuracy of the model.
    * Compare multiple logistic regression models.
### Week 2: Implementing logistic regression from scratch. ###
    * Extract features from Amazon product reviews.
    * Convert an SFrame into a NumPy array, and implement the link function for logistic regression.
    * Write a function to compute the derivative of the log likelihood function with respect to a single coefficient.
    * Write a function to compute the derivative of the log likelihood function with an L2 penalty with respect to a single coefficient.
    * Implement gradient ascent with and without an L2 penalty.
    * Compute classification accuracy for the logistic regression model.
    * Empirically explore how the L2 penalty can ameliorate overfitting.
### Week 3: Build a classification model to predict whether or not a loan provided by LendingClub is likely to default. ###
    * Use SFrames to do some feature engineering.
    * Train a decision-tree on the LendingClub dataset and visualize the tree.
    * Predict whether a loan will default along with prediction probabilities (on a validation set).
    * Train a complex tree model and compare it to a simple tree model.
    * Build a binary decision tree from scratch, and evaluate its accuracy.
        - Transform categorical variables into binary variables
        - Compute the number of mis-classified examples in an intermediate node
        - Find the best feature to split on
### Week 4: Explore various techniques for preventing overfitting in decision trees. ###
    * Implement binary decision trees with different early stopping methods.
    * Compare models with different stopping parameters.
    * Visualize the concept of overfitting in decision trees.
### Week 5: Explore the use of boosting using pre-implemented gradient boosted trees in GraphLab Create. ###
    * Train a boosted ensemble of decision-trees (gradient boosted trees) on the LendingClub loans dataset.
    * Predict whether a loan will default along with prediction probabilities (on a validation set).
    * Evaluate the trained model and compare it with a baseline.
    * Find the most positive and negative loans using the learned model.
    * Explore how the number of trees influences classification performance.
    * Implement Adaboost ensembling.
        - Modify the decision trees to incorporate weights.
        - Use your implementation of Adaboost to train a boosted decision stump ensemble.
        - Evaluate the effect of boosting (adding more decision stumps) on performance of the model.
        - Explore the robustness of Adaboost to overfitting.
### Week 6: Explore precision and recall in the context of classifiers. ###
    * Train a logistic regression model on Amazon review data.
    * Explore various evaluation metrics: accuracy, confusion matrix, precision, recall.
    * Explore how various metrics can be combined to produce a cost of making an error.
    * Explore precision and recall curves.
### Week 7: Implement a logistic regression classifier using Stochastic Gradient Ascent. ###
    * Extract features from Amazon product reviews.
    * Write a function to compute the derivative of log likelihood function with respect to a single coefficient.
    * Implement stochastic gradient ascent.
    * Compare convergence of stochastic gradient ascent with that of batch gradient ascent.

### How do I get set up? ###

* Python version: 2.7.12
* GraphLab Create: greater than 1.8.3
* Dependencies: Numpy, Matplotlib, IPython