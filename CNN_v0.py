import numpy as np
import matplotlib.pyplot as plt


# We define the activation function
def f(x, opt, a):
    # Options: 1.Sigmoid, 2.Tanh, 3.ReLU, 4.Leaky ReLU
    if opt == 1:
        return 1 / (1 + np.exp(-x))
    elif opt == 2:
        return 2 / (1 + np.exp(-2 * x)) - 1
    elif opt == 3:
        if x < 0:
            return 0
        else:
            return x
    elif opt == 4:
        if x < 0:
            return np.abs(a * x)
        else:
            return x


def df(x, opt, a):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    if opt == 1:
        aux = f(x, 1, None)
        return aux * (1 - aux)
    elif opt == 2:
        print("Not implemented")
        return None
    elif opt == 3:
        if x < 0:
            print("Not implemented")
            return None
        else:
            print("Not implemented")
            return None
    elif opt == 4:
        if x < 0:
            print("Not implemented")
            return None
        else:
            print("Not implemented")
            return None


# We define the loss function in order to train the network classifier
def loss_function(y_true, y_pred):
    # We use the mean absolute error
    return np.mean((y_true - y_pred) ** 2)


# We define the structure of our Neuronal Network
class Neural_Network:
    def __init__(self):
        self.n_weights = 6
        self.n_bias = 3
        self.learn_rate = 0.1
        self.weights = np.random.normal(0, 1, self.n_weights)
        self.bias = np.random.normal(0, 1, self.n_bias)

    def predict(self, x):
        h1 = f(np.dot(self.weights[0:2], x) + self.bias[0], 1, None)
        h2 = f(np.dot(self.weights[2:4], x) + self.bias[1], 1, None)
        o1 = f(np.dot(self.weights[4:6], np.array([h1, h2])) + self.bias[2], 1, None)
        return h1, h2, o1

    def backpropagation(self, x, y):
        h1, h2, o1 = self.predict(x)
        y_pred = o1

        # We compute all the derivatives
        dL_dy = -2 * (y - y_pred)
        do1_dh1 = self.weights[4] * df(np.dot(self.weights[4:6], np.array([h1, h2])) + self.bias[2], 1, None)
        do1_dh2 = self.weights[5] * df(np.dot(self.weights[4:6], np.array([h1, h2])) + self.bias[2], 1, None)
        do1_db3 = df(np.dot(self.weights[4:6], np.array([h1, h2])) + self.bias[2], 1, None)
        do1_dw5 = h1 * df(np.dot(self.weights[4:6], np.array([h1, h2])) + self.bias[2], 1, None)
        do1_dw6 = h2 * df(np.dot(self.weights[4:6], np.array([h1, h2])) + self.bias[2], 1, None)
        dh1_dw1 = x[0] * df(np.dot(self.weights[0:2], x) + self.bias[0], 1, None)
        dh1_dw2 = x[1] * df(np.dot(self.weights[0:2], x) + self.bias[0], 1, None)
        dh1_db1 = df(np.dot(self.weights[0:2], x) + self.bias[0], 1, None)
        dh2_dw3 = x[0] * df(np.dot(self.weights[2:4], x) + self.bias[1], 1, None)
        dh2_dw4 = x[1] * df(np.dot(self.weights[2:4], x) + self.bias[1], 1, None)
        dh2_db2 = df(np.dot(self.weights[2:4], x) + self.bias[1], 1, None)

        # We update the weights
        self.weights[0] -= self.learn_rate * dL_dy * do1_dh1 * dh1_dw1
        self.weights[1] -= self.learn_rate * dL_dy * do1_dh1 * dh1_dw2
        self.weights[2] -= self.learn_rate * dL_dy * do1_dh2 * dh2_dw3
        self.weights[3] -= self.learn_rate * dL_dy * do1_dh2 * dh2_dw4
        self.weights[4] -= self.learn_rate * dL_dy * do1_dw5
        self.weights[5] -= self.learn_rate * dL_dy * do1_dw6

        # We update the biases
        self.bias[0] -= self.learn_rate * dL_dy * do1_dh1 * dh1_db1
        self.bias[1] -= self.learn_rate * dL_dy * do1_dh2 * dh2_db2
        self.bias[2] -= self.learn_rate * dL_dy * do1_db3

    def train(self, x, y):
        n_epochs = 1000
        loss = np.zeros(n_epochs)
        for n in range(n_epochs):
            y_pred = np.apply_along_axis(self.predict, 1, x)
            loss[n] = loss_function(y, y_pred[:, 2])
            for i in range(x.shape[0]):
                self.backpropagation(x[i, :], y[i])
        plt.plot(loss, color = 'green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title("Neural Network vs. Epochs")
        plt.show()


# Define dataset: Consists in two inputs regarding height and weight (height-176)(weight-65) and we want to classify
# whether they are male or female with respecting labels 0 and 1.
data = np.array([
    [-6, -5],  # Maria
    [17, 20],  # Uri
    [11, 14],  # Jordi
    [-11, -10],  # Clara
    [-9, -9]  # Rosa
])
all_y_trues = np.array([
    1,  # Maria
    0,  # Uri
    0,  # Jordi
    1,  # Clara
    1  # Rosa
])

# We train our neuronal network
network = Neural_Network()
network.train(data, all_y_trues)

# We can use the classifier to predict new data such as,
print("The classifier predicts that Emma belongs to label %d" % (round(network.predict(np.array([-1, -1]))[2])))
print("The classifier predicts that Aleix belongs to label %d" % (round(network.predict(np.array([2, 10]))[2])))
