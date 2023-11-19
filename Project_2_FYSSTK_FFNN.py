import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_digits
import sklearn.preprocessing as skp
import scikitplot as skplot
import sklearn.metrics as skm




class FFNN:
    def __init__(self, inputs, labels, hidden_layers_sizes: list, n_categories, actfunc, epochs, batch_size, eta, lmd):

        self.X_full = inputs
        self.Y_full = labels
        self.X_data, self.Y_data = None, None

        self.n_inputs, self.n_features = self.X_full.shape
        self.hidden_layers_sizes = hidden_layers_sizes
        self.n_categories = n_categories

        self.probabilities = None
        self.a_h = []
        self.z_h = []

        # activation function
        if actfunc == 'sigmoid':
            self.actfunc = self.sigmoid
        else:
            self.actfunc = self.RELU

        # output activation function
        if n_categories > 1:
            self.out_actfunc = self.softmax
        else:
            self.out_actfunc = self.sigmoid


        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // batch_size
        self.eta = eta
        self.lmd = lmd

        self.create_weights_and_biases()

    def create_weights_and_biases(self):
        # creating weights and biases

        self.weights = []
        self.biases = []

        layer_input_size = self.n_features  # to start with the first input layer is n_features
        for layer_size in self.hidden_layers_sizes:

            hidden_weights = np.random.randn(layer_input_size, layer_size) # calculating weights from pervious layer to current layer
            hidden_bias = np.zeros(layer_size) + np.random.uniform(0.01, 0.05, layer_size)  # setting biases for current layer equal to zero plus some uniform noise
            self.weights.append(hidden_weights)
            self.biases.append(hidden_bias)
            layer_input_size = layer_size  # setting the current layer to be the previous layer

        # weights and bias in the output layer
        self.weights.append(np.random.randn(self.hidden_layers_sizes[-1], self.n_categories))
        self.biases.append(np.zeros(self.n_categories) + np.random.uniform(0.01, 0.05, self.n_categories))

    def sigmoid(self, x):
        # clip recommended by chat
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def RELU(X):
        return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))

    def softmax(self, x):
        x -= np.max(x, axis=-1, keepdims=True)
        delta = 10e-10
        return np.exp(x) / (np.sum(np.exp(x), axis=-1, keepdims=True) + delta)

    def FeedForward(self):  # for training

        # reset matrices
        self.z_h = list()
        self.a_h = list()

        a = self.X_data  # renaming the variable to remedy redefining issues we had
        self.a_h.append(a)
        for i in range(len(self.weights) - 1):
            z = np.matmul(a, self.weights[i]) + self.biases[i]  # weighted sum of inputs to the hidden layer
            a = self.sigmoid(z)  # activation function of inputs to the hidden layers and also the output layer in our case
            self.z_h.append(z)
            self.a_h.append(a)

        z_o = np.matmul(self.a_h[-1], self.weights[-1]) + self.biases[-1]
        a_o = self.out_actfunc(z_o)

        self.probabilities = a_o  # the final output


    def FeedForward_out(self, x):  # for output
        a = x
        for i in range(len(self.weights) - 1):
            z = np.matmul(a, self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)

        z_o = np.matmul(a, self.weights[-1]) + self.biases[-1]
        a_o = self.out_actfunc(z_o)

        return a_o


    def Backpropagation(self):
        if self.n_categories > 1:  # categorization
            error_output = self.probabilities - self.Y_data

        else:  # binary
            error_output = self.probabilities - self.Y_data.reshape(self.probabilities.shape)

        # Calculating output first

        hid_weight_grad = np.matmul(self.a_h[-1].T, error_output) + self.regularization(self.weights[-1])
        hid_bias_grad = np.sum(error_output, axis=0)

        # calculating error in the last hidden layer before updating weights
        error = np.matmul(error_output, self.weights[-1].T) * self.classification(self.a_h[-1])
        self.weights[-1] -= self.eta * hid_weight_grad
        self.biases[-1] -= self.eta * hid_bias_grad

        # calculating the error hidden layers
        for i in reversed(range(len(self.weights)-1)):
            hid_weight_grad = np.matmul(self.a_h[i].T, error) + self.regularization(self.weights[i])
            hid_bias_grad = np.sum(error, axis=0)
            self.weights[i] -= self.eta * hid_weight_grad
            self.biases[i] -= self.eta * hid_bias_grad
            error = np.matmul(error, self.weights[i].T)


    def regularization(self, x):
        return x * self.lmd


    def classification(self, x):

        if self.n_categories > 1:
            return x * (1 - x)
        else:
            return 1


    def predict(self, x):  # mixing a pred and a pred_prob method into one


        if self.n_categories > 1: # multiclass classification
            y_pred_prob = self.FeedForward_out(x)
            y_pred = np.argmax(y_pred_prob, axis=1)

        else:  # binary
            y_pred = self.FeedForward_out(x)
            y_pred_prob = y_pred.copy()

            for i in range(len(y_pred)):
                if y_pred[i] >= 0.5:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0

        return y_pred, y_pred_prob


    def train(self, show_progress=None):
        data_indices = np.arange(self.n_inputs)

        acc_over_training = []
        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints without replacement
                chosen_datapoints = np.random.choice(data_indices, size=self.batch_size, replace=False)

                # minibatch training data
                self.X_data = self.X_full[chosen_datapoints]
                self.Y_data = self.Y_full[chosen_datapoints]

                self.FeedForward()
                self.Backpropagation()

            '''            
            y_tilde, _ = self.predict(self.X_full)
            acc_over_training.append(skm.accuracy_score(self.Y_full, y_tilde))

        if show_progress:
            iter_list = np.arange(len(acc_over_training))
            plt.plot(iter_list, acc_over_training)
            plt.xlabel('Iteration nr.')
            plt.ylabel('Accuracy')
            plt.title('Accuracy over training')
            plt.show()
            '''



def test_hyperparameters(X_train, X_test, y_train, y_test, batch_size, epoch, actfunc, hid_layers, n_categories,N):
    # Testing hyperparameters

    eta_list = np.logspace(-3, 0, N)
    lmd_list = np.logspace(-4, -1, N)
    acc_array = np.zeros((N, N))
    for i, eta in enumerate(eta_list):
        for j, lmd in enumerate(lmd_list):
            NN_BCD_search = FFNN(X_train, y_train, hidden_layers_sizes=hid_layers, n_categories=n_categories,
                          actfunc=actfunc, epochs=epoch, batch_size=batch_size, eta=eta, lmd=lmd)
            NN_BCD_search.train()
            y_pred, y_pred_prob = NN_BCD_search.predict(X_test)
            acc_array[i][j] = skm.accuracy_score(y_test, y_pred)

    indices = np.argwhere(acc_array == np.max(acc_array))
    i, j = indices[0]

    print(f'Eta[{i}] = {eta_list[i]} and lmd[{j}] = {lmd_list[j]} gives an Acc of {acc_array[i][j]} \n')

    return acc_array, eta_list[i], lmd_list[j], eta_list, lmd_list

def test_epochs():
    epoch_list = np.arange(0, 500, 15)
    acc_list = []
    for i in range(len(epoch_list)):
        NN_BCD = FFNN(X_train, y_train, hidden_layers_sizes=hid_layers, n_categories=1,
                      epochs=150, batch_size=100, eta=eta, lmd=lmd)
        NN_BCD.train()
        y_pred, y_pred_prob = NN_BCD.predict(X_test)
        acc_list.append(skm.accuracy_score(y_test, y_pred))
    idx = np.argmax(acc_list)
    print(f'Epoch = {epoch_list[idx]} gives best acc = {acc_list[idx]}')

    return epoch_list[idx]

def param_heatmap(acc_array):
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(acc_array, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

def conf_mat(y_test, y_pred):
    skplot.metrics.plot_confusion_matrix(y_test, y_pred)
    plt.show()


# function taken from Week 42 in lecture notes
def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    n_categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, n_categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector


if __name__ == '__main__':
    np.random.seed(2018)

    # import data
    #data = load_breast_cancer()
    data = load_digits()
    X, y = data.data, data.target

    # scale the data
    scaler = skp.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
    y_train_onehot, y_test_onehot = to_categorical_numpy(y_train), to_categorical_numpy(y_test)

    # define the structure of our hidden layers for different problems
    hid_layers_WBC = [10, 3]
    hid_layers_MNIST = [300, 300]

    # testing hyperparameters
    acc_array, eta, lmd, eta_list, lmd_list = test_hyperparameters(X_train, X_test, y_train_onehot, y_test, actfunc='RELU', batch_size=20,
                                               epoch=50, N=10, n_categories=10, hid_layers=hid_layers_MNIST)
    param_heatmap(acc_array)

    eta_WBC, lmd_WBC = 1.0, 0.00046415888336127773   # found from test_hyperparameter function
    eta_MNIST, lmd_MNIST = 0.1, 0.002154434690031882

    # finding best num of epochs
    #epoch = test_epochs()

    #print(eta, lmd, epoch)
    # running algorithm with optimal parameters
    NN_BCD = FFNN(inputs=X_train, labels=y_train_onehot, hidden_layers_sizes=hid_layers_MNIST, n_categories=10,
                  actfunc='RELU', epochs=50, batch_size=20, eta=eta, lmd=lmd)
    NN_BCD.train()
    y_pred, y_pred_prob = NN_BCD.predict(X_test)

    # confusion matrix
    conf_mat(y_test, y_pred)

    # checking performance
    #print(f'MSE = {skm.mean_squared_error(y_test, y_pred_prob)}')
    print(f'Accuracy = {skm.accuracy_score(y_test, y_pred) * 100:.1f}%')



