def StochasticGradientDescent(X, y, learning_rate, batch_size, max_iter, cost_limit):
    """
    The Stochastic method for Gradient Descent Algorithm
    """
    import numpy as np
    def linear_fit(X, w):
        return np.dot(X, w)

    def gradient(X, y, w):
        y_pred = linear_fit(X, w)
        g = np.dot(X.T, y_pred-y)
        return g

    def cost_func(X, y, w):
        y_pred = linear_fit(X, w)
        cost = ((np.dot((y_pred-y).T, y_pred-y))/2)[0][0]
        return cost

    def create_batches(X, y, batch_size):
        data_mat = np.hstack((X, y))
        np.random.shuffle(data_mat)
        if data_mat.shape[0] < batch_size:
            print("WARNING! Not enough data to fill the batch. All data has been used.")
            batch = (data_mat[:,:-1], data_mat[:,-1].reshape((-1,1)))
        else:
            batch = (data_mat[:batch_size,:-1], data_mat[:batch_size,-1].reshape((-1,1)))
        return batch

    w = np.zeros((X.shape[1],1))
    cost = cost_limit + 1
    i = 0
    while i < max_iter and cost > cost_limit:
        batch = create_batches(X, y, batch_size)
        for xi, yi in zip(batch[0], batch[1]):
            xi, yi = xi.reshape((1, -1)), yi.reshape((-1,1))
            w = w - learning_rate * gradient(xi, yi, w)
            cost = cost_func(xi, yi, w)
        i += 1
    return w, cost, i
