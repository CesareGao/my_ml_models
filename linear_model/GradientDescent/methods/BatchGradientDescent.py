def BatchGradientDescent(X, y, learning_rate, max_iter, cost_limit):
    import numpy as np
    def linear_fit(X, w):
        return np.dot(X, w)

    def gradient(X, y, w):
        y_pred = linear_fit(X, w)
        g = np.dot(X.T, y_pred-y)
        return g

    def cost_func(X, y, w):
        y_pred = linear_fit(X, w)
        cost = 1/X.shape[0]*((np.dot((y_pred-y).T, y_pred-y))/2)[0][0]
        return cost

    w = np.zeros((X.shape[1],1))
    cost = cost_limit + 1
    i = 0
    while i < max_iter and cost > cost_limit:
        w = w - learning_rate * (1/X.shape[0]) * gradient(X, y, w)
        cost = cost_func(X, y, w)
        i += 1
    return w, cost, i
