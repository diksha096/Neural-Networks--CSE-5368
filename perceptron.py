# Sharma, Diksha
# 1001-679-176
# 2019-09-22
# Assignment-01-01

import numpy as np
import itertools

class Perceptron(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights =np.random.randn(self.number_of_classes,self.input_dimensions+1)
        return None
        
        raise Warning("You must implement _initialize_weights! This function should initialize (or re-initialize) your model weights. Bias should be included in the weights")

    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros((self.number_of_classes,self.input_dimensions+1))
        return None
        raise Warning("You must implement this function! This function should initialize (or re-initialize) your model weights to zeros. Bias should be included in the weights")

    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]
        """
        n,m=X.shape
        X0 = np.ones((1,m))
        print("xx",X0)
        Xnew=np.vstack((X0,X))
        print(Xnew)
        Y=np.dot(self.weights,Xnew)
        output=np.where(Y<0,0,1)
        return output 
        raise Warning("You must implement predict. This function should make a prediction on a matrix of inputs")


    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print("weights",self.weights)
        raise Warning("You must implement print_weights")
        
    "add ones to the input samples"
    def add_ones(self, T):      
        one = np.ones((1,T.shape[1]))
        final = np.vstack([one, T])
        return final
    
    def train(self, X, Y, num_epochs=10, alpha=0.001):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :return: None
        """
        print("x",X)
        print("y",Y)
        for i in range(0,num_epochs):
            with_ones = self.add_ones(X)          
            for input_val in range(0,with_ones.shape[1]):
                main_op = np.dot(self.weights,with_ones[:,[input_val]])       
                main_op = np.where(main_op<0, 0, 1)                
                error = np.subtract(Y[:,[input_val]],main_op)
                self.weights[:,1:] = self.weights[:,1:] + alpha*( np.dot(error,with_ones[1:,[input_val]].transpose()))
                self.weights[:,[0]] = self.weights[:,[0]] + alpha*error
        return None
        raise Warning("You must implement train")

    def calculate_percent_error(self,X, Y):
        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the output is not hte same as the desired output, Y,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [number_of_classes ,n_samples]
        :return percent_error
        """
        num_of_errors = 0 
        with_ones=self.add_ones(X)  
        output_pred = np.dot(self.weights, with_ones)        
        output_pred = np.where(output_pred<0, 0, 1)
        for i in range(with_ones.shape[1]):
            if list(output_pred[:,i]) != list(Y[:,i]) : 
                num_of_errors=num_of_errors+1
        percent_error=num_of_errors/with_ones.shape[1]
        return percent_error
        raise Warning("You must implement calculate_percent_error")

if __name__ == "__main__":
    """
    This main program is a sample of how to run your program.
    You may modify this main program as you desire.
    """    
    input_dimensions = 2
    number_of_classes = 2
    
    model = Perceptron(input_dimensions=input_dimensions, number_of_classes=number_of_classes, seed=1)
    X_train = np.array([[-1.43815556, 0.10089809, -1.25432937, 1.48410426],
                       [-1.81784194, 0.42935033, -1.2806198, 0.06527391]])
    print(model.predict(X_train))
    Y_train = np.array([[1, 0, 0, 1], [0, 1, 1, 0]])
    model.initialize_all_weights_to_zeros()
    print("****** Model weights ******\n",model.weights)
    print("****** Input samples ******\n",X_train)
    print("****** Desired Output ******\n",Y_train)
    percent_error=[]
    for k in range (20):
        model.train(X_train, Y_train, num_epochs=1, alpha=0.0001)
        percent_error.append(model.calculate_percent_error(X_train,Y_train))
    print("******  Percent Error ******\n",percent_error)
    print("****** Model weights ******\n",model.weights)