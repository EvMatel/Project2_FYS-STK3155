import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from sklearn.model_selection import train_test_split
from imageio import imread
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skl
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns



def sigmoid(x):
    return 1 / (1 + np.exp(x))


def CostOLS(y,X,theta):
    return np.sum((y-X @ theta)**2)


def CostCross(y, X, theta):
    m = len(y)
    return -(1.0/m)*np.sum(y*np.log(X@theta+10e-10))


def GradDesc(X_train, Y_train, momentum, n_iter, n_epochs, eta, cost, use_ADAM=0, use_grad=0, use_RMSprop=0, lmd=0.01):
    n = len(X_train)
    # run the stochastic gradient descent
    if use_grad == 1:
        training_grad = grad(cost, 2)

    # sett up for momentum
    change = 0.0
    delta_momentum = momentum

    if use_RMSprop == 1:
        rho = 0.99
        delta = 1e-7
    if use_ADAM == 1:
        theta = 0.001 * np.random.randn(len(X_train[0]), 1)
        beta1 = 0.9
        beta2 = 0.999
        delta = 1e-7
        iter = 0
        for epoch in range(n_epochs):
            first_moment = 0.0
            second_moment = 0.0
            iter += 1
            for i in range(n_iter):
                gradient = training_grad(Y_train, X_train, theta) + (lmd*theta)
                first_moment = beta1 * first_moment + (1 - beta1) * gradient
                second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
                first_term = first_moment / (1.0 - beta1 ** iter)
                second_term = second_moment / (1.0 - beta2 ** iter)
                update = eta * first_term / (np.sqrt(second_term) + delta)
                theta -= update

    else:
        theta = 0.01 * np.random.randn(len(X_train[0]), 1)
        for epochs in range(n_epochs):
            Giter = 0.0
            for i in range(n_iter):
                gradient = training_grad(Y_train, X_train, theta)
                if use_grad == 1:
                    new_change = eta * gradient + delta_momentum * change
                    theta -= new_change
                    change = new_change
                if use_RMSprop == 1:
                    Giter = rho * Giter + (1 - rho) * gradient * gradient
                    update = gradient * eta / delta + np.sqrt(Giter)
                    theta -= update

    return theta


def S_grad_descent(X_train, Y_train, momentum, minibatch_size, n_epochs, eta, cost, use_ADAM=0, use_grad=0, use_RMSprop=0, lmd=0.01, clip_gradients=True):

    n = len(X_train)
    M = minibatch_size
    m = int(n/M)  # number of minibatches

    # run the stochastic gradient descent
    if use_grad == 1:
        training_grad = grad(cost, 2)

    # sett up for momentum
    change = 0.0
    delta_momentum = momentum


    if use_RMSprop == 1:
        rho = 0.99
        delta = 1e-7
    if use_ADAM == 1:
        theta = 0.0001 * np.random.randn(len(X_train[0]), 1)
        beta1 = 0.9
        beta2 = 0.999
        delta = 1e-5
        iter = 0
        data_indices = np.arange(len(X_train))
        for epoch in range(n_epochs):
            first_moment = 0.0
            second_moment = 0.0
            iter += 1
            for i in range(m):
                choose_datapoints = np.random.choice(data_indices, size=minibatch_size, replace=False)
                xi = X_train[choose_datapoints]
                yi = Y_train[choose_datapoints]
                gradient = (1.0/M)*training_grad(yi, xi, theta) + (lmd * theta)
                if clip_gradients:
                    gradient = np.clip(gradient, -1, 1)
                first_moment = beta1*first_moment+(1-beta1)*gradient
                second_moment = beta2*second_moment+(1-beta2)*gradient*gradient
                first_term = first_moment/(1.0-beta1**iter)
                second_term = second_moment/(1.0-beta2**iter)
                update = eta*first_term/(np.sqrt(second_term)+delta)
                theta -= update

    else:
        theta = 0.01 * np.random.randn(len(X_train[0]), 1)
        data_indices = np.arange(len(X_train))
        for epochs in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                choose_datapoints = np.random.choice(data_indices, size=minibatch_size, replace=False)
                xi = X_train[choose_datapoints]
                yi = Y_train[choose_datapoints]
                gradient = (1.0/M)*training_grad(yi, xi, theta)+(lmd*theta)
                if use_grad == 1:
                    new_change = eta*gradient+delta_momentum*change
                    theta -= new_change
                    change = new_change
                if use_RMSprop == 1:
                    Giter = rho*Giter+(1-rho)*gradient*gradient
                    update = gradient*eta/delta+np.sqrt(Giter)
                    theta -= update

    return theta


def MSE(y_data, y_model):
   return skl.mean_squared_error(y_data, y_model)


def Accuracy(y_test, y_pred):
    return skl.accuracy_score(y_test, y_pred)


# from project 1
def MSE(y_data, y_model):
   n = np.size(y_model)
   return np.sum((y_data - y_model) ** 2) / n


def sigmoid(x):
    return 1 / (1 + np.exp(x))


def OLS(theta,  X_train, X_test):
    OLS_ztilde = X_train @ theta
    OLS_zpredict = X_test @ theta
    for i in range(len(X_test)):
        if OLS_zpredict[i] > 0.5:
            OLS_zpredict[i] = 1
        else:
            OLS_zpredict[i] = 0

    return OLS_ztilde, OLS_zpredict, theta



if __name__ == '__main__':
    np.random.seed(2018)
    data = load_breast_cancer()
    X, y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    y_train = y_train.reshape((len(y_train), 1))

    theta_s = S_grad_descent(X_train, y_train, momentum=0.8, minibatch_size=50, n_epochs=100, eta=1e-4, cost=CostOLS,
                           use_grad=1, use_ADAM=1, lmd=1e-3)
    _, OLS_zpredict_s, _ = OLS(theta_s, X_train, X_test)
    print(MSE(y_test, OLS_zpredict_s))
    print(Accuracy(y_test, OLS_zpredict_s))
    theta_g = GradDesc(X_train, y_train, momentum=0.8, n_iter=50, n_epochs=100, eta=1.5e-3,
                                               cost=CostOLS, use_grad=1, use_ADAM=1, lmd=1e-2)

    _, OLS_zpredict_g, _ = OLS(theta_g, X_train, X_test)
    print(MSE(y_test, OLS_zpredict_s))
    print(Accuracy(y_test, OLS_zpredict_s))

    cm = confusion_matrix(y_test, OLS_zpredict_s)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('confusion_M_GD')
    plt.show()
    """

    """
    sns.set()
    eta_g = [1e-1, 1e-2, 1.5e-2, 1e-3, 1.5e-3, 1e-4]
    lmd_g = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    test_accuracy = np.zeros((len(eta_g), len(lmd_g)))

    for i in range(len(eta_g)):
        for j in range(len(lmd_g)):
            theta = GradDesc(X_train, y_train, momentum=0.8, n_iter=50, n_epochs=100, eta=eta_g[i],
                                   cost=CostOLS, use_grad=1, use_ADAM=1, lmd=lmd_g[j])
            _, OLS_zpredict, _ = OLS(theta, X_train, X_test)
            test_accuracy[i][j] = Accuracy(y_test, OLS_zpredict)

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    fig.savefig('parameter_testing_GD')
    plt.show()




