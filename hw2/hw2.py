import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
                 0.25: 1.32,
                 0.1: 2.71,
                 0.05: 3.84,
                 0.0001: 100000},
             2: {0.5: 1.39,
                 0.25: 2.77,
                 0.1: 4.60,
                 0.05: 5.99,
                 0.0001: 100000},
             3: {0.5: 2.37,
                 0.25: 4.11,
                 0.1: 6.25,
                 0.05: 7.82,
                 0.0001: 100000},
             4: {0.5: 3.36,
                 0.25: 5.38,
                 0.1: 7.78,
                 0.05: 9.49,
                 0.0001: 100000},
             5: {0.5: 4.35,
                 0.25: 6.63,
                 0.1: 9.24,
                 0.05: 11.07,
                 0.0001: 100000},
             6: {0.5: 5.35,
                 0.25: 7.84,
                 0.1: 10.64,
                 0.05: 12.59,
                 0.0001: 100000},
             7: {0.5: 6.35,
                 0.25: 9.04,
                 0.1: 12.01,
                 0.05: 14.07,
                 0.0001: 100000},
             8: {0.5: 7.34,
                 0.25: 10.22,
                 0.1: 13.36,
                 0.05: 15.51,
                 0.0001: 100000},
             9: {0.5: 8.34,
                 0.25: 11.39,
                 0.1: 14.68,
                 0.05: 16.92,
                 0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    total_instances = len(data)
    labels = np.unique(data[:, -1])
    for label in labels:
        count = len(data[data[:, -1] == label])
        gini += (count / total_instances) ** 2
    gini = 1 - gini
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    total_instances = len(data)
    labels = np.unique(data[:, data.shape[1] - 1])
    for label in labels:
        count = len(data[data[:, data.shape[1] - 1] == label])
        div = count / total_instances
        entropy += div * np.log2(div)
    entropy = -entropy
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {}  # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    split_information = 1
    if gain_ratio:
        impurity_func = calc_entropy
        split_information = calc_entropy(data[:, feature].reshape(-1, 1))

    impurity_func_sum = 0.0
    total_instances = len(data)
    values = np.unique(data[:, feature])
    for value in values:
        groups[value] = data[data[:, feature] == value]
        count = len(groups[value])
        impurity_func_sum += (count / total_instances) * impurity_func(groups[value])
    information_gain = impurity_func(data) - impurity_func_sum

    if split_information == 0:
        groups = {}
    else:
        goodness = information_gain/split_information
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        max_count = 0
        values = np.unique(self.data[:, -1])
        for value in values:
            count = len(self.data[self.data[:, -1] == value])
            if count > max_count:
                max_count = count
                pred = value
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)

    def split(self, impurity_func):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        max_goodness = 0
        best_feature = -1
        best_groups = {}
        
        for feature in range(self.data.shape[1] - 1):
            goodness, groups = goodness_of_split(self.data, feature, impurity_func, self.gain_ratio)
            if goodness > max_goodness:
                max_goodness = goodness
                best_feature = feature
                best_groups = groups
        self.feature = best_feature
        
        if impurity_func(self.data) == 0 or self.depth == self.max_depth or self.test_chi_square():
            self.terminal = True
            return
        
        for value in best_groups.keys():
            node = DecisionNode(data=best_groups[value], feature=self.feature, depth=self.depth + 1, chi=self.chi,
                                max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(node, value)
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################

    def test_chi_square(self):
        """
        Tests if splitting according to the chosen feature gives a distribution
        which is similar to exactly random.

        Output: True if the split distribution is random and False otherwise.
        """
        if self.chi == 1:
            return False
        
        chi_square_statistic = 0
        labels = np.unique(self.data[:, -1])
        values = np.unique(self.data[:, self.feature])
        
        for value in values:
            d_f = len(self.data[self.data[:, self.feature] == value])
            for label in labels:
                p_label = len(self.data[self.data[:, -1] == label]) / len(self.data)
                p_f = len(self.data[(self.data[:, self.feature] == value) & (self.data[:, -1] == label)])
                e_f = d_f * p_label
                chi_square_statistic += ((p_f - e_f) ** 2 / e_f)
        
        degree_of_freedom = (len(labels) - 1) * (len(values) - 1)
        return chi_square_statistic <= chi_table[degree_of_freedom][self.chi]


def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset.
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    root = DecisionNode(data=data, chi=chi, max_depth=max_depth, gain_ratio=gain_ratio)
    nodes = [root]
    while len(nodes) > 0:
        node = nodes.pop()
        node.split(impurity)
        nodes.extend(node.children)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root


def predict(root, instance):
    """
    Predict a given instance using the decision tree
 
    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.
 
    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    node = root
    while not node.terminal:
        value = instance[node.feature]
        if value in node.children_values:
            index_of_feature_node = node.children_values.index(value)
            node = node.children[index_of_feature_node]
        else:
            break
    pred = node.pred
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return pred


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy
 
    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated
 
    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for instance in dataset:
        result_of_predict = predict(node, instance)
        if result_of_predict == instance[-1]:
            accuracy += 1

    accuracy = (accuracy / len(dataset)) * 100
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, max_depth=max_depth)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def chi_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    p_values = [1, 0.5, 0.25, 0.1, 0.05, 0.0001]
    for p_value in p_values:
        tree = build_tree(data=X_train, impurity=calc_entropy, gain_ratio=True, chi=p_value)
        chi_training_acc.append(calc_accuracy(tree, X_train))
        chi_testing_acc.append(calc_accuracy(tree, X_test))
        depth.append(calc_tree_depth(tree))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth


def calc_tree_depth(node):
    """
    Calculates the depth of the given node.

    Input:
    - node: the root of the tree.

    Output: the depth of the tree.
    """
    if node.terminal:
        return 0
    else:
        max_child_depth = 0
        for child in node.children:
            child_depth = calc_tree_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        return max_child_depth + 1


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of nodes in the tree.
    """
    n_nodes = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    if node.terminal:
        return 1

    n_nodes = 1
    for child in node.children:
        n_nodes += count_nodes(child)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes
