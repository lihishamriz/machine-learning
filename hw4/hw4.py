import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import norm


# The compute_cost function calculates the cost of the
# logistic regression model. It takes two arguments:
# sigmoidVector, which represents the predicted probabilities,
# and y, which contains the true labels.
# It computes the cost using the logistic
# regression cost function and returns the result.

def compute_cost(sigmoidVector, y):
    return (- np.dot(y, np.log(sigmoidVector)) - np.dot((1 - y), np.log(1 - sigmoidVector))) / len(y)


# The sigmoid function implements the sigmoid
# activation function, which is used to
# transform the input values into
# probabilities between 0 and 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # set random seed
        np.random.seed(self.random_state)

        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.theta = np.random.random(size=(len(X.T) + 1))
        X = np.column_stack((np.array([1] * len(X)), X))  # Bias trick
        iterations = 0
        while iterations < self.n_iter and (
                iterations <= 1 or abs(self.Js[iterations - 1] - self.Js[iterations - 2]) > self.eps):
            sigmoidVector = sigmoid(X @ self.theta)
            self.Js.append(compute_cost(sigmoidVector, y))
            outputMinusTargetVector = sigmoidVector - y
            self.theta -= self.eta * (X.T @ outputMinusTargetVector)
            iterations += 1
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        preds = []
        X = np.column_stack((np.array([1] * len(X)), X))  # Bias trick
        for x in X:
            probability = sigmoid(np.dot(x, self.theta))
            prediction = 1 if probability >= 0.5 else 0
            preds.append(prediction)
        preds = np.array(preds)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, folds)
    accuracies = []
    for fold in fold_indices:
        train_indices = np.setdiff1d(indices, fold)  # Split the data into training and validation sets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[fold], y[fold]
        model = algo
        model.fit(X_train, y_train)  # Train the algorithm on the training set
        y_pred = model.predict(X_val)  # Make predictions on the validation set
        accuracy = np.mean(y_pred == y_val)  # Calculate accuracy for the fold
        accuracies.append(accuracy)
    cv_accuracy = np.mean(accuracies)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if np.isscalar(sigma):
        if sigma == 0:
            p = np.where(data == mu, 1, 0)
        else:
            p = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    else:
        p = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((data - mu) / sigma) ** 2)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return p


class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = None

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.weights = np.ones(self.k) / self.k
        indices = np.random.choice(data.shape[0], self.k, replace=False)
        self.mus = data[indices]
        self.sigmas = np.ones(self.k) * np.std(data)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        # probabilities = self.weights * norm_pdf(data[:, None], self.mus, self.sigmas)
        # # Normalize the responsibilities
        # self.responsibilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        weights = np.array(self.weights)
        mus = np.array(self.mus)
        sigmas = np.array(self.sigmas)

        self.responsibilities = weights[None, :] * norm_pdf(data[:, None], mus[None, :], sigmas[None, :])
        self.responsibilities = self.responsibilities / self.responsibilities.sum(axis=1, keepdims=True)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        N = data.shape[0]
        self.weights = np.sum(self.responsibilities, axis=0) / N
        self.mus = np.sum(self.responsibilities * data[:, None], axis=0) / (N * self.weights)
        self.sigmas = np.sqrt(np.sum(self.responsibilities * (data[:, None] - self.mus) ** 2, axis=0) / (
                N * self.weights))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    def compute_cost_EM(self, data):
        N = len(data)
        K = self.k
        total_costs = 0
        for k in range(K):
            total_costs += -N * np.log(self.weights[k]) + np.sum(np.log(norm_pdf(data, self.mus[k], self.sigmas[k])))
        return total_costs / K

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        self.init_params(data)

        self.costs = []

        for iteration in range(self.n_iter):
            self.expectation(data)
            self.maximization(data)

            cost = self.compute_cost_EM(data)
            self.costs.append(cost)

            if iteration > 0 and np.all(abs(cost - self.costs[iteration - 1]) < self.eps):
                break
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas


def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    K = len(weights)  # Number of Gaussians
    pdf = np.zeros_like(data, dtype=float)

    for k in range(K):
        pdf += weights[k] * norm.pdf(data, mus[k], sigmas[k])
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pdf


class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.em_models = dict()
        self.k = k
        self.random_state = random_state
        self.prior = dict()

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels = np.unique(y)
        for label in labels:
            data = X[y == label]  # Select data for this class
            em = EM(self.k, random_state=self.random_state)  # Create EM instance for this class
            em.fit(data)  # Fit EM model on the data of this class
            self.em_models[label] = em  # Store the fitted model
            self.prior[label] = len(data) / len(X)  # Estimate prior probability for this class
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        labels = list(self.prior.keys())  # Get the list of class labels
        log_likelihood = np.zeros((len(X), len(labels)))  # Initialize log-likelihood array

        for i, label in enumerate(labels):
            weights, mus, sigmas = self.em_models[label].get_dist_params()
            log_likelihood[:, i] = np.log(self.prior[label]) + np.log(gmm_pdf(X, weights, mus, sigmas)).sum(axis=1)

        labels = np.array(list(self.prior.keys()))  # Get the list of class labels
        preds = labels[np.argmax(log_likelihood, axis=1)]  # Predict the class with the highest log-likelihood
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return preds


def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    '''

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    lor = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    lor.fit(x_train, y_train)
    
    lor_train_acc = get_accuracy(y_train, lor.predict(x_train))
    lor_test_acc = get_accuracy(y_test, lor.predict(x_test))
    
    naive_bayes = NaiveBayesGaussian(k)
    naive_bayes.fit(x_train, y_train)
    
    bayes_train_acc = get_accuracy(y_train, naive_bayes.predict(x_train))
    bayes_test_acc = get_accuracy(y_test, naive_bayes.predict(x_test))
    
    plot_decision_regions(x_train, y_train, lor, title="Logistic Regression")
    plot_decision_regions(x_train, y_train, naive_bayes, title="Naive Bayes")
    
    plt.plot(range(len(lor.Js)), lor.Js)
    plt.title("Logistic Regression Cost")
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}


def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    mean = np.array([1, 1, 1])
    cov_class0 = np.array([[4, 0, 0], [0, 4, 0], [0, 0, 4]])

    class0 = np.random.multivariate_normal(mean, cov_class0, 250)
    class0_labels = np.zeros(250)

    class1 = np.random.rand(250, 3)
    class1_labels = np.ones(250)

    dataset_a_features = np.append(class0, class1, axis=0)
    dataset_a_labels = np.append(class0_labels, class1_labels, axis=0)
    dataset_a_with_labels = np.insert(dataset_a_features, 3, dataset_a_labels, axis=1)
    np.random.shuffle(dataset_a_with_labels)

    fig = plt.figure(figsize=(5, 5))
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.scatter(class0[:, 0], class0[:, 1], class0[:, 2], marker='o')
    ax1.scatter(class1[:, 0], class1[:, 1], class1[:, 2], marker='o')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.set_zlabel('Feature 3')
    plt.show()


    new_class0_feature1 = np.random.random(250)
    new_class0_feature2 = new_class0_feature1 * 4
    new_class0_feature3 = new_class0_feature1 * 6
    new_class0 = np.column_stack((new_class0_feature1, new_class0_feature2))
    new_class0 = np.column_stack((new_class0, new_class0_feature3))

    new_class1_feature1 = np.random.random(250)
    new_class1_feature2 = new_class1_feature1 * 3
    new_class1_feature3 = new_class1_feature1 * 4
    new_class1 = np.column_stack((new_class1_feature1, new_class1_feature2))
    new_class1 = np.column_stack((new_class1, new_class1_feature3))

    dataset_b_features = np.append(new_class0, new_class1, axis=0)
    dataset_b_labels = np.append(np.zeros(250), np.ones(250), axis=0)
    dataset_b_with_labels = np.insert(dataset_b_features, 3, dataset_b_labels, axis=1)
    np.random.shuffle(dataset_b_with_labels)
    
    fig = plt.figure(figsize=(5, 5))
    ax2 = fig.add_subplot(111, projection='3d')
    ax2.scatter(new_class0[:, 0], new_class0[:, 1], new_class0[:, 2], marker='o')
    ax2.scatter(new_class1[:, 0], new_class1[:, 1], new_class1[:, 2], marker='o')
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_zlabel('Feature 3')
    plt.show()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return {'dataset_a_features': dataset_a_features,
            'dataset_a_labels': dataset_a_labels,
            'dataset_b_features': dataset_b_features,
            'dataset_b_labels': dataset_b_labels
            }


def get_accuracy(data, pred):
    return np.mean(data == pred)

# Function for ploting the decision boundaries of a model
def plot_decision_regions(X, y, classifier, resolution=0.01, title=""):

    # setup marker generator and color map
    markers = ('.', '.')
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.title(title)
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
    plt.show()
