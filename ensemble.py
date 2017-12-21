import pickle
import numpy as np

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.classifiers_ = []
        self.alpha_ = []
        pass

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''

        num_samples, num_features = X.shape
        # Initial weights the sample weights are initialized to 1 / num_samples
        sample_weight = np.ones(num_samples) * (1 / num_samples)

        for iboost in range(self.n_weakers_limit):
            estimator = self.weak_classifier
            estimator.fit(X, y, sample_weight=sample_weight)
            print("Finish " + str(iboost + 1) + "th weak classifier.")
            self.classifiers_.append(estimator)
            y_predict = estimator.predict(X)
            y_predict = y_predict.reshape((len(y_predict), 1))

            incorrect = y_predict != y
            # Error fraction
            estimator_error = np.sum(sample_weight.dot(incorrect))

            # Stop if classification is perfect
            if estimator_error <= 0:
                return self

            alpha = np.log(1. / (estimator_error) - 1) / 2.
            self.alpha_.append(alpha)

            for j in range(len(sample_weight)):
                sample_weight[j] = sample_weight[j] * np.exp(-y[j]*alpha*y_predict[j])
            sample_weight /= np.sum(sample_weight)
        return self

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        pass

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        pred = sum(classifier.predict(X).T * w
                   for classifier, w in zip(self.classifiers_,
                                           self.alpha_))
        pred = np.sign(pred)
        return pred

    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
