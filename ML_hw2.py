import numpy as np

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return x * (1.0 - x)

#def linear(x):

#def linear_derivative(x):

class NeuralNetwork:
    def __init__(self, x, y, k):
        self.input = x
        self.k = k
        self.y = y
        self.output = np.zeros(self.y.shape)
        #first layer has whatever dimension of x input
        #make it go to k[0] neurons
        weights1 = np.random.rand(self.input.shape[1],self.k[0])

        #weights list
        self.weights = []
        self.biases = []
        self.weights.append(weights1)
        for i in range(0,len(self.k) - 1):
            input_neurons = self.k[i]
            output_neurons = self.k[i+1]
            current_weights = np.random.rand(input_neurons, output_neurons)
            self.weights.append(current_weights)
            bias_vector = np.random.rand(len(x), 1)
            bias_matrix = []
            for i in range(0, self.k[i+1]):
                bias_matrix.append(bias_vector)
            
            self.biases.append(np.array(bias_matrix).T)

    def feedforward(self):
        #first layer
        self.layer1 = sigmoid(np.dot(self.input, self.weights[0]))
        #make layers array
        self.layers = []
        self.layers.append(self.layer1)
        #rest of the layers
        for i in range(0, len(self.k) - 1):
            next_layer = sigmoid(np.dot(self.layers[i], self.weights[i+1])\
                + self.biases[i][0])
            self.layers.append(next_layer)

    def backprop(self):
        for i in self.layers[::-1]:
            pass
        # d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output)\
        #     * sigmoid_derivative(self.output)))
        # d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output)\
        #     * sigmoid_derivative(self.output), self.weights2.T) 
        #     * sigmoid_derivative(self.layer1)))
        # l = len(self.layers)
        # d_weights_layerl = np.dot(self.layers[l-1].T, 
        #     (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        # d_weights = []
        #for i in range(): #go from layer l-1 to 0

        #self.weights1 += d_weights1
        #self.weights2 += d_weights2

if __name__ == "__main__":
    X = np.array([[0,0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])

    y = np.array([[0],[1],[1],[0]])
    k= [3,6,7,3]

    nn = NeuralNetwork(X,y,k)

    nn.feedforward()
    print(nn.output)
    # for i in range(10):
        #nn.backprop()

