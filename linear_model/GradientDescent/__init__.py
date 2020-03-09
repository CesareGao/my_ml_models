class GradientDescent:
    def __init__(self, method="minibatch"):
        if method == "batch":
            from .methods.BatchGradientDescent import BatchGradientDescent
            self.algorithm = BatchGradientDescent()
        elif method == "sgd":
            from .methods.StochasticGradientDescent import StochasticGradientDescent
            self.algorithm = StochasticGradientDescent()
        elif method == "minibatch":
            from .methods.MiniBatchGradientDescent import MiniBatchGradientDescent
            self.algorithm = MiniBatchGradientDescent()
        else:
            raise ValueError(f"No such a method exists: {method}")    
