from numpy import *
import scipy.io
from scipy import optimize
import sys

# sigmoid(x) = 1/(1+exp[-x]), use scipy implementation.
sigmoid = scipy.special.expit

def sigmoid_gradient(x):
    """sigmoid'(x) = sigmoid(x) * [1-sigmoid(x)] = 1/(4 cosh[x/2]^2)"""
    return 0.25 / (cosh(0.5 * x) ** 2)

class SingleLayerNeuralNetwork:
    m: float
    number_input_layers: int=-1
    number_hidden_layers: int=-1
    number_output_layers: int=-1
    theta_size: int=-1 # Total size of weight vector.
    theta_slice_index: int=-1 # Index to slice Theta in Theta1 and Theta2

    def __init__(self, lambd=0.0):
        self._lambda = lambd

    def set_training_data(self, X, y):
        self.m, self.number_input_layers = X.shape
        self.X = insert(X, 0, 1.0, 1) # Insert ones before first column
        self.y = y

    def set_neural_network_size(self, hidden_layer_size, output_layer_size):
        self.number_hidden_layers = hidden_layer_size
        self.number_output_layers = output_layer_size
        self.theta_slice_index = (self.number_input_layers + 1) * self.number_hidden_layers
        self.size_Theta1 = self.number_hidden_layers * (self.number_input_layers+1)
        self.size_Theta2 = (self.number_hidden_layers+1) * self.number_output_layers
        self.theta_size = self.size_Theta1 + self.size_Theta2

    def slice_theta(self, Theta):
        """Seperate Theta in Theta1 and Theta2"""
        Th1_arr, Th2_arr = (Theta[0:self.theta_slice_index], Theta[self.theta_slice_index:])
        Theta1 = Th1_arr.reshape((self.number_hidden_layers, self.number_input_layers + 1))
        Theta2 = Th2_arr.reshape((self.number_output_layers, self.number_hidden_layers + 1))
        return (Theta1, Theta2)

    def cost(self, Theta):
        """Vectorised version of cost function"""
        Theta1, Theta2 = self.slice_theta(Theta)
        h1 = sigmoid(self.X @ Theta1.transpose())
        h2 = sigmoid(insert(h1, 0, 1.0, 1) @ Theta2.transpose())
        Y = zeros([Theta2.shape[0], len(self.y)])
        Y[(self.y-1).flatten(), range(self.m)] = 1.0

        J = (trace(-Y @ log(h2) - (1 - Y) @ log(1 - h2))) / self.m

        # Regulariser contribution, omit first column
        Th1Sqr = (Theta1[:,1:])**2
        Th2Sqr = (Theta2[:,1:])**2
        J += self._lambda/(2.0 * self.m) * (sum(Th1Sqr) + sum(Th2Sqr))
        return J

    def backprop(self, Theta):
        """Vectorised back propagation algorithm. Can probably be optimised further."""
        Theta1, Theta2 = self.slice_theta(Theta)
        
        Theta1_grad = zeros(Theta1.shape)
        Theta2_grad = zeros(Theta2.shape)  # K-by-hiddenunits+1

        z2 = self.X @ Theta1.transpose()  # m-by-hiddenunits
        h1 = sigmoid(z2)  # m-by-hidden-units
        h1 = insert(h1, 0, 1.0, 1)
        h2 = sigmoid(h1 @ Theta2.transpose())
        Y = zeros([Theta2.shape[0], len(self.y)])
        Y[(self.y - 1).flatten(), range(self.m)] = 1.0
        Y = Y.transpose()  # optimise later

        delta_3 = h2 - Y  # m-by-K matrix
        Theta2_grad = (delta_3.transpose() @ h1) / self.m
        tussen_result = sigmoid_gradient(z2)
        delta_2 = (delta_3 @ Theta2[:, 1:]) * tussen_result
        Theta1_grad = delta_2.transpose() @ self.X / self.m

        # Regulariser contribution for gradient
        Theta1_grad[:, 1:] += self._lambda / self.m * Theta1[:, 1:]
        Theta2_grad[:, 1:] += self._lambda / self.m * Theta2[:, 1:]
        return append(Theta1_grad.flatten(), Theta2_grad.flatten())
        
    def predict(self, Theta):
            """Predict number using weights"""
            Theta1, Theta2 = self.slice_theta(Theta)
            h1 = sigmoid(self.X @ Theta1.transpose())
            h2 = sigmoid(insert(h1, 0, 1.0, 1) @ Theta2.transpose())
            lijst = argmax(h2, axis=1) + 1
            return lijst

    def train(self, eps_init=0.12):
        # Initialise weights of neural network
        Theta0 = random.random(self.theta_size)*2*eps_init - eps_init
        output = optimize.fmin_cg(self.cost, Theta0, self.backprop, maxiter=50, full_output=True, disp=True)
        self.Theta = output[0]
        


mat = scipy.io.loadmat("ex4weights.mat")
mat2 = scipy.io.loadmat("ex4data1.mat")
X = mat2['X']
y = mat2['y']
Theta1 = mat['Theta1']
Theta2 = mat['Theta2']
Theta = append(Theta1.flatten(), Theta2.flatten())
_lambda = 1.0
num_labels=10

simple_network = SingleLayerNeuralNetwork(_lambda)
simple_network.set_training_data(X,y)
simple_network.set_neural_network_size(hidden_layer_size=25, output_layer_size=num_labels)

sys.stdout.write("Training neural network...\n")
sys.stdout.flush()

simple_network.train()

predictions = simple_network.predict(Theta)
accuracy = average(y.flatten() == predictions)
print("Accuracy: {}".format(accuracy))
