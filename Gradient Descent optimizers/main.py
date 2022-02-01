import numpy as np


class Regressor:

    def __init__(self) -> None:
        self.optimizer = 'adamax'
        # str(input())
        self.X, self.y = self.generate_dataset(n_samples=200, n_features=1)
        n, d = self.X.shape
        self.w = np.zeros((d, 1))

    def generate_dataset(self, n_samples, n_features):
        """
        Generates a regression dataset
        Returns:
            X: a numpy.ndarray of shape (100, 2) containing the dataset
            y: a numpy.ndarray of shape (100, 1) containing the labels
        """
        from sklearn.datasets import make_regression

        np.random.seed(42)
        X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=30)
        y = y.reshape(n_samples, 1)
        return X, y

    def linear_regression(self):
        """
        Performs linear regression on a dataset
        Returns:
            y: a numpy.ndarray of shape (n, 1) containing the predictions
        """
        y = np.dot(self.X, self.w)
        return y

    def predict(self, X):
        """
        Predicts the labels for a given dataset
        X: a numpy.ndarray of shape (n, d) containing the dataset
        Returns:
            y: a numpy.ndarray of shape (n,) containing the predictions
        """
        y = np.dot(X, self.w).reshape(X.shape[0])
        return y

    def compute_loss(self):
        """
        Computes the MSE loss of a prediction
        Returns:
            loss: the loss of the prediction
        """
        predictions = self.linear_regression()
        loss = np.mean((predictions - self.y) ** 2)
        return loss

    def compute_gradient(self):
        """
        Computes the gradient of the MSE loss
        Returns:
            grad: the gradient of the loss with respect to w
        """
        predictions = self.linear_regression()
        dif = (predictions - self.y)
        grad = 2 * np.dot(self.X.T, dif)
        return grad

    def fit(self, n_iters=100, render_animation=False):
        """
        Trains the model
        optimizer: the optimization algorithm to use
        X: a numpy.ndarray of shape (n, d) containing the dataset
        y: a numpy.ndarray of shape (n, 1) containing the labels
        weights : list of all weights that change across of operations
        loss : list of all loss that changes across of operations
        vt : momentum vt
        gradient_sum : sum of all gradients
        E_grad : rmsprop E_grad
        n_iters: the number of iterations to train for
        """
        figs = []
        weights = []
        loss = []
        vt = 0
        gradient_sum = 0
        E_grad = 0
        m = 0
        v = 0
        for i in range(1, n_iters + 1):
            if self.optimizer == 'gd':
                self.gradient_descent(0.001)
                pass

            elif self.optimizer == "sgd":
                self.sgd_optimizer(0.01)

                pass

            elif self.optimizer == "sgdMomentum":
                vt = self.sgd_momentum(0.01, 0.9, vt)

                weights.append(self.w)
                loss.append(self.compute_loss().tolist())

                pass

            elif self.optimizer == "adagrad":
                weights.append(self.w)
                self.adagrad_optimizer(gradient_sum, 2, 500, weights)
                gradient_sum += self.compute_gradient() ** 2

                pass

            elif self.optimizer == "rmsprop":
                weights.append(self.w)

                E_grad = self.rmsprop_optimizer(gradient_sum, 0.01, 0.9, 0.0001, E_grad)
                gradient_sum += self.compute_gradient() ** 2

                pass

            elif self.optimizer == "adam":
                weights.append(self.w)

                m, v = self.adam_optimizer(m, v, 1, 0.9, 0.7, 0.01, iter_num=i)

                pass

            elif self.optimizer == "adamax":
                m, v = self.Adamax(m, v, 0.7, 0.9, 0.9, 0.000001, iter_num=i)

                pass

            if i % 10 == 0:
                print("Iteration: ", i)
                print("Loss: ", self.compute_loss())

            if render_animation:
                import matplotlib.pyplot as plt
                from moviepy.video.io.bindings import mplfig_to_npimage

                fig = plt.figure()
                plt.scatter(self.X, self.y, color='red')
                plt.plot(self.X, self.predict(self.X), color='blue')
                plt.xlim(self.X.min(), self.X.max())
                plt.ylim(self.y.min(), self.y.max())
                plt.title(f'Optimizer:{self.optimizer}\nIteration: {i}')
                plt.close()
                figs.append(mplfig_to_npimage(fig))

        if render_animation and len(figs) > 0:
            from moviepy.editor import ImageSequenceClip
            clip = ImageSequenceClip(figs, fps=5)
            clip.write_gif(f'{self.optimizer}_animation.gif', fps=5)

    def gradient_descent(self, alpha):
        """
        Performs gradient descent to optimize the weights
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        self.w = self.w - alpha * self.compute_gradient()
        return self.w

    def sgd_optimizer(self, alpha):
        """
        Performs stochastic gradient descent to optimize the weights'
        alpha: the learning rate
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        for i in range(200):
            index = np.random.randint(100)
            index1 = np.random.randint(200)
            xi = self.X[index:index1]
            yi = self.y[index:index1]
            n = len(xi)
            for j in range(n):
                self.w = self.w - alpha * ((1 / n) * (self.predict(xi[j]) - yi[j]) * xi[j])
        return self.w

    def sgd_momentum(self, alpha=0.01, momentum=0.9, vt=0):
        """
        Performs SGD with momentum to optimize the weights'
        alpha: the learning rate
        momentum: the momentum
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
        """
        gradient = self.compute_gradient()
        vt = momentum * vt + alpha * gradient
        self.w = self.w - vt
        return vt

    def adagrad_optimizer(self, g, alpha, epsilon, weights):
        """
        Performs Adagrad optimization to optimize the weights'
        alpha: the learning rate
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        gradients = self.compute_gradient()
        gradient_update = gradients / (np.sqrt(g )+ epsilon)

        self.w = self.w - (gradient_update * alpha)
        return self.w

    def rmsprop_optimizer(self, g, alpha, beta, epsilon, E_grad2):
        """
        Performs RMSProp optimization to optimize the weights
        g: sum of squared gradients'
        alpha: the learning rate
        beta: the momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        gradients = self.compute_gradient()

        E_grad2 = (beta * E_grad2) + ((1 - beta) * (gradients * gradients))

        self.w = self.w - ((alpha / (np.sqrt(E_grad2)) + epsilon) * gradients)

        return E_grad2

    def adam_optimizer(self, m, v, alpha, beta1, beta2, epsilon, iter_num):
        """
        Performs Adam optimization to optimize the weights
        m: the first moment vector
        v: the second moment vector
        alpha: the learning rate
        beta1: the first momentum
        beta2: the second momentum
        epsilon: a small number to avoid division by zero
        Returns:
            w: a numpy.ndarray of shape (d, 1) containing the optimized weights
            ...
        """
        gradients = self.compute_gradient()

        m = beta1 * m + (1. - beta1) * gradients

        v = beta2 * v + (1. - beta2) * gradients ** 2

        mt_hat = m / (1. - beta1 ** (iter_num + 1))

        vt_hat = v / (1. - beta2 ** (iter_num + 1))

        self.w = self.w - (mt_hat * alpha / (np.sqrt(vt_hat) + epsilon))

        return m, v

    def Adamax(self, mt, vt, alpha, beta1, beta2, epsilon, iter_num):

        gradients = self.compute_gradient()

        mt = beta1 * mt + (1. - beta1) * gradients

        vt = np.maximum(beta2 * vt, np.abs(gradients))

        mt_hat = mt / (1. - beta1 ** (iter_num + 1))

        self.w = self.w - ((alpha / (vt + epsilon)) * mt_hat)

        return mt, vt


if __name__ == "__main__":
    A = Regressor()
    print(A.w)
    print(A.w)
    print(A.X.shape)

    A.fit(render_animation=True)
    # A.plot_gradient(10,10)
