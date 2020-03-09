def MiniBatchGradientDescent(X, y, learning_rate, batch_size, max_iter, cost_limit):
    """
    The minibatch method for Gradient Descent Algorithm
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
        cost = 1/X.shape[0]*((np.dot((y_pred-y).T, y_pred-y))/2)[0][0]
        return cost

    def create_batches(X, y, batch_size):
        batches = []
        data_mat = np.hstack((X, y))
        np.random.shuffle(data_mat)
        if (data_mat.shape[0] % batch_size) == 0:
            n_batch = (data_mat.shape[0]) // batch_size
        else:
            n_batch = (data_mat.shape[0]) // batch_size + 1
        for i in range(n_batch):
            batch = data_mat[i*batch_size: (i+1)*batch_size, :]
            batch_x = batch[:,:-1]
            batch_y = batch[:,-1].reshape((-1,1))
            batches.append((batch_x, batch_y))
        return batches

    w = np.zeros((X.shape[1],1))
    cost = 1
    i = 0
    while i < max_iter and cost > cost_limit:
        print(f"round {i}")
        batches = create_batches(X, y, batch_size)
        for batch in batches:
            w = w - learning_rate * (1/batch[0].shape[0]) * gradient(batch[0], batch[1], w)
            cost = cost_func(batch[0], batch[1], w)
        i += 1
    return w, cost
