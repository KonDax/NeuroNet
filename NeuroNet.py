import numpy as np
import random
import openpyxl as xl


class OurNeuralNetwork(object):

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def derivative_sigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, show_tests=False):
        if test_data:
            n_test = len(test_data)
            mx_evol = -1
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                evol = self.evaluate(test_data, self.biases, self.weights)
                if mx_evol < evol:
                    mx_biases = self.biases
                    mx_weights = self.weights
                    mx_evol = evol
                if show_tests: print("Epoch {0}: {1} / {2}".format(j, evol, n_test))
            else:
                print("Epoch {0} complete".format(j))

        if test_data:
            print("\nAll Epoches complete \nBest score: {0} / {1}".format(
                mx_evol, n_test))
            self.biases = mx_biases
            self.weights = mx_weights

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = self.cost_derivative(activations[-1], y) * \
            self.derivative_sigmoid(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.derivative_sigmoid(z)
            delta = np.dot(np.transpose(self.weights[-l+1]), delta) * sp
            nabla_b[-l] = delta
            #print(delta, "\n\n", np.transpose(activations[-l+1]))
            nabla_w[-l] = np.dot(delta, np.transpose(activations[-l+1]))
        return (nabla_b, nabla_w)

    def evaluate(self, test_data, biases, weights):
        test_results = [(np.argmax(self.feedforward(x, biases, weights)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def feedforward(self, a, biases, weights):
        for b, w in zip(biases, weights):
            a = self.sigmoid(np.dot(w, a)+b)
        return a

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


if __name__ == "__main__":
    Network = OurNeuralNetwork([12, 7, 1])
    wb = xl.load_workbook("./Тестовые данные.xlsx")
    data = [([wb["Лист1"].cell(row=i, column=j).value for j in range(2, 14)], wb["Лист1"][f"N{i}"].value)
         for i in range(2, 100)]
    Network.SGD(data, 1000, 7, 2, data, True)
