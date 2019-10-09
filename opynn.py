import numpy as np

######################################
### Functions
######################################


def kalasfunktion():
    print('KALAS!')
    
def create_data_batches(X, labels, batch_size, replace = False):
    mini_batches = []
    data = np.c_[X, labels]
    n_batches = data.shape[0] // batch_size
    np.random.shuffle(data)
    for i in range(n_batches + 1):
        mini_batch = data[i*batch_size:(i+1)*batch_size,:]
        mini_batches.append(mini_batch)
    return mini_batches

def stochastic_gradient_descent(network, data, train_rate):
    x = data[:,0:-1]
    t = data[:,-1]
    n_train = t.shape[0]
    random_pattern = np.random.randint(n_train)

    network.feed_forward(x[random_pattern,:])
    network.backpropagation(t[random_pattern])
    network.stochastic_gradient_descent(train_rate)

def mini_batch_gradient_descent(network, data_batches, train_rate):
    n_batches = len(data_batches)
    n_layers = len(network.layers)
    for i_batch in range(n_batches):
        x = data[i_batch][:, 0:-1]
        t = data[i_batch][:, -1]
        batch_size = t.shape[0]
        
        # Feed forward through the network
        v = [network.feed_forward(x[i,:]) for i in range(batch_size)]
        
        # Backpropagate
        error = [network.backpropagation(t[i]) for i in range(batch_size)]
        
        # Update weights        
        for i_layer in range(n_layers):
            for i_data in range(batch_size):
                network.layers[i_layer].stochastic_gradient_descent(train_rate, error[i_data][i_layer], v[i_data][i_layer])

                

############################################################
### Layer classes
############################################################
                
class FeedForwardLayer:

    def __init__(self, n_input, layer_size, activation_function = 'linear', rand_range = (-1,1)):
        self.weights = ( rand_range[1] - rand_range[0] ) * np.random.random((layer_size,n_input)) + rand_range[0]
        self.thresholds = np.zeros((layer_size,))
        self.activation_function = activation_function;
        self.layer_dimensions = (n_input, layer_size)
        
    def set_weights(self, w):
        if np.shape(w) == np.shape(self.weights):
            self.weights = w
        else:
            print('Warning: Weight matrix dimensions not compliant, dimensions of the layer are',  np.shape(self.weights))
        
    def set_thresholds(self,t):
        
        if np.shape(t) == np.shape(self.thresholds):
            self.thresholds = t
        else:
            print('Warning: Threshold dimensions not compliant, dimensions of the thresholds are',  np.shape(self.thresholds))
       
    def feed_forward(self, layer_input):
        self.b = - self.thresholds + self.weights @ layer_input
        
        # Select activation function
        if self.activation_function.lower() == 'linear':           
            return self.linear()
        elif self.activation_function.lower() == 'relu':
            return self.relu()
        elif self.activation_function.lower() == 'heaviside':
            return self.heaviside()
        elif self.activation_function.lower() == 'signum':
            return self.signum()
        elif self.activation_function.lower() == 'sigmoid':
            return self.sigmoid()
        elif self.activation_function.lower() == 'tanh':
            return self.tanh()
        elif self.activation_function.lower() == 'softmax':
            return self.softmax()
        else:
            print(activation_function, 'is not an avalible activation function')
            
        
    def output_error(self, train_label):
        if self.activation_function == '':
            print('Error: Layer output has not yet been computed. Try running FeedForwardLayer.feed_forward first.')
            return None
        else:
            self.propagation_error = (train_label - self.output) * self.dg
            return self.propagation_error
            
    def backpropagation(self, next_layer_error, next_layer_weights):
        if self.activation_function == '':
            print('Error: Layer output has not yet been computed. Cannot backpropagate. Try running FeedForwardLayer.feed_forward first.')
            return None
        else:
            self.propagation_error = np.dot(next_layer_error, next_layer_weights) * self.dg
            return self.propagation_error
        
        
    ###############################
    ### Optimization Algorithms ###
    ###############################
    
    def stochastic_gradient_descent(self, train_rate, prop_error, layer_input):
        self.weights = self.weights + train_rate * np.outer(prop_error, layer_input)
        self.thresholds = self.thresholds - train_rate * prop_error
  

    ############################
    ### Activation functions ###
    ############################
    
    def linear(self):
        # Returns layer output without activation function
        self.activation_function = 'linear'
        self.dg = np.ones(np.size(self.thresholds))
        self.output = self.b
        return self.output
    
    def relu(self):
        # Rectified linear unit function
        self.activation_function = 'relu'
        self.output = np.maximum(0,self.b)
        self.dg = np.sign(self.output)       
        return self.output

    def heaviside(self):
        # Heavyside step function
        self.activation_function = 'heaviside'
        self.output = np.heaviside(self.b,0)
        self.dg = np.zeros(np.size(self.thresholds))        
        return self.output

    def signum(self):
        self.activation_function = 'signum'
        self.output = np.sign(self.b)        
        self.dg = np.zeros(np.size(self.thresholds))
        return self.output
    
    def sigmoid(self):
        self.activation_function = 'sigmoid'
        self.output = 1 / (1 + np.exp(-self.b))
        self.dg = np.exp(-self.b)/(1 + np.exp(-self.b))**2        
        return self.output

    def tanh(self):
        self.activation_function = 'tanh'
        self.output = np.tanh(self.b)
        self.dg = 1 - self.output**2
        return self.output
    
    # Not tested yet
    def softmax(self, alpha = 1):
        self.activation_function = 'softmax'
        self.output = np.exp(alpha*self.b)/np.sum(np.exp(alpha*self.b))
        self.dg = 1
        return self.output
    
    
    #############################
    ### Information Retrieval ###
    #############################
    
    def get_weights(self):
        return self.weights
    
    def get_thresholds(self):
        return self.thresholds
    
    def shape(self):
        return self.layer_dimensions
    
    def tpe(self):
        return 'Fully Connected Layer'
    
    
    
##########################################
### Network class
##########################################
    
class NeuralNetwork:
    
    def __init__(self, layer_list):
        self.layers = layer_list
        self.network_depth = len(layer_list)
        self.v = [np.zeros((self.layers[0].shape()[0]))]
        self.error = []
        for i in range(self.network_depth):
            self.v.append(np.zeros((self.layers[i].shape()[1])))
            self.error.append(np.zeros((self.layers[i].shape()[1])))
        
    
    def output(self, data):
        self.v[0] = data.copy() # This is probabliy not the optimal way in order to implement mini batch later    
        for i in range(self.network_depth):
            self.v[i+1] = self.layers[i].feed_forward(self.v[i])
        return self.v[-1]
    
    def feed_forward(self, data):
        self.v[0] = data.copy() # This is probabliy not the optimal way in order to implement mini batch later    
        for i in range(self.network_depth):
            self.v[i+1] = self.layers[i].feed_forward(self.v[i])
        return self.v
    
    def backpropagation(self, train_label):
        self.error[-1] = self.layers[-1].output_error(train_label) 
        for i in reversed(range(0,self.network_depth - 1)):    
            self.error[i] = self.layers[i].backpropagation(self.error[i+1], self.layers[i+1].get_weights())
        return self.error
        
    ### Optimization
    # This should probably be a separate function
    def stochastic_gradient_descent(self, train_rate):
        for i in range(0,self.network_depth):
            self.layers[i].stochastic_gradient_descent(train_rate, self.error[i], self.v[i])
