
from sklearn.base import BaseEstimator, ClassifierMixin
from  sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels

from ..base import ArtificialNeuralNetwork as BaseANN


class ArtificialNeuralNetwork(BaseANN, BaseEstimator, ClassifierMixin):
    def __init__(self):
       # BaseANN.__init__()
       super().__init__() 

    def fit(self, X, y, epochs, batch_size=1, parallel_computing=True, shuffle_data=True):
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        self.classes_ = unique_labels(y)
        self.train(X=X, y=y, epochs=epochs, batch_size=batch_size, parallel_computing=parallel_computing, shuffle_data=shuffle_data)
        return self

    def predict(self):
        check_is_fitted(self, ['X_', 'y_'])
        return self.forward(x)


