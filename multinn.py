# Sharma, Diksha
# 1001-679-176
# 2019-10-27
# Assignment-03-01

# using tensorflow_version 2.x
import tensorflow as tf
import numpy as np


class MultiNN(object):
    def __init__(self, input_dimension):
        """
        Initialize multi-layer neural network
        :param input_dimension: The number of dimensions for each the input data sample
        """
        self.input_dimension = input_dimension
        self.weights = []
        self.biases = []
        self.activations = []
        self.loss = None

    def add_layer(self, num_nodes, activation_function):
        self.activations.append(activation_function)      
        w_init = tf.random.normal(shape=(self.input_dimension, num_nodes), dtype='float32')
        newWeights = tf.Variable(w_init,trainable=True)
        self.weights.append(newWeights)  
        zero_bias=tf.Variable(tf.zeros(shape=(num_nodes,)))
        self.biases.append(zero_bias)
        self.input_dimension = num_nodes
         
        """
         This function adds a dense layer to the neural network
         :param num_nodes: number of nodes in the layer
         :param activation_function: Activation function for the layer
         :return: None
         """
             
         

    def get_weights_without_biases(self, layer_number):
        return self.weights[layer_number]
        """
        This function should return the weight matrix (without biases) for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0.
         :return: Weight matrix for the given layer (not including the biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         """

    def get_biases(self, layer_number):
        return self.biases[layer_number]
        """
        This function should return the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param layer_number: Layer number starting from layer 0
         :return: Weight matrix for the given layer (not including the biases). Note that the biases shape should be [1][number_of_nodes]
         """

    def set_weights_without_biases(self, weights, layer_number):
        self.weights[layer_number]=weights
        
        """
        This function sets the weight matrix for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
         :param weights: weight matrix (without biases). Note that the shape of the weight matrix should be
          [input_dimensions][number of nodes]
         :param layer_number: Layer number starting from layer 0
         :return: none
         """

    def set_biases(self, biases, layer_number):
        self.biases[layer_number]=biases
 
        """
        This function sets the biases for layer layer_number.
        layer numbers start from zero.
        This means that the first layer with activation function is layer zero
        :param biases: biases. Note that the biases shape should be [1][number_of_nodes]
        :param layer_number: Layer number starting from layer 0
        :return: none
        """

    def set_loss_function(self, loss_fn):
        """
        This function sets the loss function.
        :param loss_fn: Loss function
        :return: none
        """
        self.loss = loss_fn

    def sigmoid(self, x):

        return tf.nn.sigmoid(x)

    def linear(self, x):
        return x

    def relu(self, x):
        out = tf.nn.relu(x)
        return out

    def cross_entropy_loss(self, y, y_hat):
        """
        This function calculates the cross entropy loss
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
        :param y_hat: Array of actual outputs values [n_samples][number_of_classes].
        :return: loss
        """
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=y_hat))

    def predict(self, X):
        X_copy=X
        for values in range(len(self.weights)):
            values_for_layers_pred=self.activations[values]((tf.matmul(X_copy,self.weights[values]))+self.biases[values])    
            X_copy=values_for_layers_pred
        
        return X_copy
        """
        Given array of inputs, this function calculates the output of the multi-layer network.
        :param X: Array of input [n_samples,input_dimensions].
        :return: Array of outputs [n_samples,number_of_classes ]. This array is a numerical array.
        """

    def train(self, X_train, y_train, batch_size, num_epochs, alpha=0.8, regularization_coeff=1e-6):
        input_data=tf.data.Dataset.from_tensor_slices((X_train, y_train))
        input_data=input_data.batch(batch_size)
        for epoch in range(num_epochs):
         for count_step,(x,y) in enumerate(input_data):
            with tf.GradientTape(persistent=True) as data_grad_tape:
                predicted_y_val=self.predict(x)
                loss=self.cross_entropy_loss(y,predicted_y_val)
                for weights_val in range(len(self.weights)):
                    partial_loss,partial_wb=data_grad_tape.gradient(loss,[self.weights[weights_val],self.biases[weights_val]])
                    self.weights[weights_val].assign_sub(alpha*partial_loss)
                    self.biases[weights_val].assign_sub(alpha*partial_wb)
                    
                    
        
        """
         Given a batch of data, and the necessary hyperparameters,
         this function trains the neural network by adjusting the weights and biases of all the layers.
         :param X: Array of input [n_samples,input_dimensions]
         :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
         the desired (true) class.
         :param batch_size: number of samples in a batch
         :param num_epochs: Number of times training should be repeated over all input data
         :param alpha: Learning rate
         :param regularization_coeff: regularization coefficient
         :return: None
         """

    def calculate_percent_error(self, X, y):
        from sklearn.metrics import accuracy_score
        final_predictions=[]
        predictions_Class=self.predict(X)
        for predictions in predictions_Class.numpy():  
            final_predictions.append(np.argmax(predictions))
        percent_error=round(1 - accuracy_score(list(y),final_predictions),3)
        return percent_error
            
        
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """

    def calculate_confusion_matrix(self, X, y):
        from sklearn.metrics import confusion_matrix
        final_predictions=[]
        predictions_Class=self.predict(X)
        for predictions in predictions_Class.numpy():  
            final_predictions.append(np.argmax(predictions))
        cmatrix=confusion_matrix(y,final_predictions)
        return cmatrix
        """
        Given input samples and corresponding desired (true) output as indexes,
        this method calculates the confusion matrix.
        :param X: Array of input [n_samples,input_dimensions]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """


if __name__ == "__main__":
    from tensorflow.keras.datasets import mnist

    np.random.seed(seed=1)
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # Reshape and Normalize data
    X_train = X_train.reshape(-1, 784).astype(np.float64) / 255.0 - 0.5
    y_train = y_train.flatten().astype(np.int32)
    input_dimension = X_train.shape[1]
    indices = list(range(X_train.shape[0]))
    # np.random.shuffle(indices)
    number_of_samples_to_use = 500
    X_train = X_train[indices[:number_of_samples_to_use]]
    y_train = y_train[indices[:number_of_samples_to_use]]
    multi_nn = MultiNN(input_dimension)
    number_of_classes = 10
    activations_list = [multi_nn.sigmoid, multi_nn.sigmoid, multi_nn.linear]
    number_of_neurons_list = [50, 20, number_of_classes]
    for layer_number in range(len(activations_list)):
        multi_nn.add_layer(number_of_neurons_list[layer_number], activation_function=activations_list[layer_number])
    np.random.seed(seed=1)
    for layer_number in range(len(multi_nn.weights)):
        W = multi_nn.get_weights_without_biases(layer_number)
        W = tf.Variable((np.random.randn(*W.shape)) * 0.1, trainable=True)
        multi_nn.set_weights_without_biases(W, layer_number)
        b = multi_nn.get_biases(layer_number=layer_number)
        b = tf.Variable(np.zeros(b.shape) * 0, trainable=True)
        multi_nn.set_biases(b, layer_number)
    multi_nn.set_loss_function(multi_nn.cross_entropy_loss)
    percent_error = []
    for k in range(10):
        multi_nn.train(X_train, y_train, batch_size=100, num_epochs=20, alpha=0.8)
        percent_error.append(multi_nn.calculate_percent_error(X_train, y_train))
    confusion_matrix = multi_nn.calculate_confusion_matrix(X_train, y_train)
    print("Percent error: ", np.array2string(np.array(percent_error), separator=","))
    print("************* Confusion Matrix ***************\n", np.array2string(confusion_matrix, separator=","))
