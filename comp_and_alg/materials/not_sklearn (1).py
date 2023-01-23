import numpy as np


class SimpleModel:

    def __init__(self):
        """This should initiliase your algorithm. Here we do not need to pass any parameters.
        But we want to define a new attribute `most_common` that will contain the most common class.
        
        For initialisation, just assign it to None.
        """

        pass

    def fit(self, X, y):
        """Fit the model using X and y data. Both X and y are assumed to be numpy arrays.

        This method should find the most frequent value in y and set the attribute 
        self.most_common to it.
        There are multiple ways to do so, no need to find the most optimised one.
        """

        pass

    def predict(self, X):
        """Generates y_pred from a given X matrix. Here we want to predict self.most_common for each
        observation, so you only need to return a vector of same length as number of observations in X
        where each value is equal to self.most_common
        """

        pass


class LogisticRegression:

    def __init__(self, gamma=0.001, nr_steps=1000):
        """Initialise your model with `w` as None. Our algorithm also takes two
        parameters, the learning rate gamma (which we default to .001 here) and the
        number of steps nr_steps (defaulting to 1000)
        """

        pass

    def sigmoid(self, y):
        """Implement a sigmoid method that computes the sigmoid for a vector
        y and returns it.
        """

        pass

    def predicted_values(self, X, w):
        """Implement the predicted_values method that takes:
        * the data `X` - an NxD dimensional matrix of the data (N datapoints)
        * a vector `w` - a D dimensional vector of the parameters

        and returns:
        * `p` - an N dimensional output of predicted probabilities
        (you can use your `sigmoid` method)
        """

        pass

    def gradient(self, X, y, w):
        """Implement the gradient method that takes:
        * `X` - an NxD dimensional matrix of the data (N datapoints)
        * `y` - a N dimensional vector containing the true labels
        * `w` - a D dimensional vector of the parameters

        and returns:
        * a vector, the gradient of the cross entropy loss.

        It's the same as for the logistic regression notebook, 
        just make sure you're reusing the predicted_values method we've defined on the class.
        """

        pass

    def fit(self, X, y):
        """Implement the fit function that takes a matrix X and a vector y as parameters.
        It should initialise self.w to a vector of zeros with the right dimension, 

        Then make self.nr_steps iterations of gradient descent to update self.w

        It is similar to what the simpleGD function from the previous notebook does,
        but here we do not need to keep the history, only setting w.
        """

        pass

    def predict(self, X):
        """Implement predict to predict binary classes, 0 or 1. 
        It takes a matrix X as input.
        It needs to use self.predicted value with X and the current 
        set of weights, self.w. Check if the probability is higher than
        a given threshold (use .5) and return the right classes accordingly.
        """

        pass