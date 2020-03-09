class GradientDescent:
    def __init__(self, method="minibatch", learning_rate=0.001, max_iter=1000, cost_limit=0.01, batch_size=None):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.method = method
        self.cost_limit = cost_limit
        if method == "batch":
            from .methods.BatchGradientDescent import BatchGradientDescent as algorithm
        elif method == "sgd":
            from .methods.StochasticGradientDescent import StochasticGradientDescent as algorithm
        elif method == "minibatch":
            if batch_size:
                self.batch_size = batch_size
            else:
                self.batch_size = 128
                print("WARNING! No batch size given. Use default size 128")
            from .methods.MiniBatchGradientDescent import MiniBatchGradientDescent as algorithm
        else:
            raise ValueError(f"No such a method exists: {method}")
        self.algorithm = algorithm

    def fit(self, X, y):
        if self.method == "minibatch":
            args = {"learning_rate": self.learning_rate,
                    "max_iter": self.max_iter,
                    "batch_size": self.batch_size,
                    "cost_limit": self.cost_limit}
        elif self.method == "batch":
            args = {"learning_rate": self.learning_rate,
                    "max_iter": self.max_iter,
                    "cost_limit": self.cost_limit}
        else:
            args = {"learning_rate": self.learning_rate,
                    "max_iter": self.max_iter,
                    "cost_limit": self.cost_limit}
        self.weights, self.mse = self.algorithm(X, y.to_numpy().reshape((-1,1)), **args)

    def predict(self, X):
        import numpy as np
        return np.dot(X, self.weights)
