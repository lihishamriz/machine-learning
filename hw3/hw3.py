import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.25,
            (0, 1): 0.25,
            (1, 0): 0.25,
            (1, 1): 0.25
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.125,
            (0, 0, 1): 0.125,
            (0, 1, 0): 0.125,
            (0, 1, 1): 0.125,
            (1, 0, 0): 0.125,
            (1, 0, 1): 0.125,
            (1, 1, 0): 0.125,
            (1, 1, 1): 0.125,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are dependent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for (x, y) in X_Y.keys():
            if not np.isclose(X[x] * Y[y], X_Y[(x, y)]):
                return True
        
        return False
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are independent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        for (x, y, c) in X_Y_C.keys():
            if not np.isclose(X_C[(x, c)] * Y_C[(y, c)] / C[c], X_Y_C[(x, y, c)]):
                return False
        
        return True
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    log_p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    factorials = [np.math.factorial(value) for value in k]
    log_p = np.sum(k * np.log(rate) - rate - np.log(factorials))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    likelihoods = []
    for rate in rates:
        likelihoods.append(poisson_log_pmf(samples, rate))
    likelihoods = np.array(likelihoods)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    rate = 0.0
    likelihoods = get_poisson_log_likelihoods(samples, rates)  # might help
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    max_rate_index = np.argmax(likelihoods)
    rate = rates[max_rate_index]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return rate

def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    mean = np.sum(samples) / len(samples)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal density function for a given x, mean and standard deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std: The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    numerator = np.exp(-(x - mean) ** 2 / (2 * (std ** 2)))
    denominator = np.sqrt(2 * np.pi * (std ** 2))
    p = numerator / denominator
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditional normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = dataset
        self.class_value = class_value
        self.data_subset = dataset[dataset[:, -1] == self.class_value]
        self.mean = np.mean(self.data_subset, axis=0)
        self.std = np.std(self.data_subset, axis=0)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = len(self.data_subset) / len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood probability of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = 1
        for j, value in enumerate(x[:-1]):
            likelihood *= normal_pdf(value, self.mean[j], self.std[j])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MAPClassifier():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object containing the relevant parameters and methods
                     for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods
                     for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        pred = 0 if posterior0 > posterior1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of predicting the class for each instance in the test-set.
        
    Output
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    predictions = [map_classifier.predict(x) for x in test_set]
    acc = np.sum(predictions == test_set[:, -1]) / len(test_set)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal density function for a given x, mean and covariance matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    x_features = x[:-1]
    d = len(x_features)
    numerator = np.exp(-0.5 * np.matmul((x_features - mean).T, np.matmul(np.linalg.inv(cov), x_features - mean)))
    denominator = np.sqrt((2 * np.pi) ** d * (np.linalg.det(cov)))
    pdf = numerator / denominator
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditional multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = dataset
        self.class_value = class_value
        self.data_subset = dataset[dataset[:, -1] == self.class_value]
        self.mean = np.mean(self.data_subset, axis=0)[:-1]
        self.cov = np.cov(self.data_subset.T[:-1])
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        
    def get_prior(self):
        """
        Returns the prior probability of the class according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = len(self.data_subset) / len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior

class MaxPrior():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object containing the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior0 = self.ccd0.get_prior()
        posterior1 = self.ccd1.get_prior()
        pred = 0 if posterior0 > posterior1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

class MaxLikelihood():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object containing the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior0 = self.ccd0.get_instance_likelihood(x)
        posterior1 = self.ccd1.get_instance_likelihood(x)
        pred = 0 if posterior0 > posterior1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilities for a discrete naive bayes
        distribution for a specific class. The probabilities are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.dataset = dataset
        self.class_value = class_value
        self.data_subset = dataset[dataset[:, -1] == self.class_value]
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    def get_prior(self):
        """
        Returns the prior probability of the class
        according to the dataset distribution.
        """
        prior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        prior = len(self.data_subset) / len(self.dataset)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        likelihood = 1
        n_i = len(self.data_subset)
        v_j = len(np.unique(x[:-1]))
        for j, value in enumerate(x[:-1]):
            n_i_j = len(self.data_subset[self.data_subset[:, j] == x[j]])
            if n_i_j == 0:
                likelihood *= EPSILLON
            else:
                likelihood *= (n_i_j + 1) / (n_i + v_j)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior probability of the instance
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0, ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object containing the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object containing the relevant parameters and methods for the distribution of class 1.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.ccd0 = ccd0
        self.ccd1 = ccd1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        posterior0 = self.ccd0.get_instance_posterior(x)
        posterior1 = self.ccd1.get_instance_posterior(x)
        pred = 0 if posterior0 > posterior1 else 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a test-set using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Output
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        predictions = [self.predict(x) for x in test_set]
        acc = np.sum(predictions == test_set[:, -1]) / len(test_set)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return acc
