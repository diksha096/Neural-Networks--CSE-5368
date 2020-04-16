# Sharma, Diksha
# 1001-679-176
# 2019-10-06
# Assignment-02-01

import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf


def display_images(images):
        # This function displays images on a grid.
        # Farhad Kamangar Sept. 2019
        number_of_images=images.shape[0]
        number_of_rows_for_subplot=int(np.sqrt(number_of_images))
        number_of_columns_for_subplot=int(np.ceil(number_of_images/number_of_rows_for_subplot))
        for k in range(number_of_images):
                plt.subplot(number_of_rows_for_subplot,number_of_columns_for_subplot,k+1)
                plt.imshow(images[k], cmap=plt.get_cmap('gray'))
                # plt.imshow(images[k], cmap=pyplot.get_cmap('gray'))
        plt.show()

def display_numpy_array_as_table(input_array):
        # This function displays a 1d or 2d numpy array (matrix).
        # Farhad Kamangar Sept. 2019
        if input_array.ndim==1:
                num_of_columns,=input_array.shape
                temp_matrix=input_array.reshape((1, num_of_columns))
        elif input_array.ndim>2:
                print("Input matrix dimension is greater than 2. Can not display as table")
                return
        else:
                temp_matrix=input_array
        number_of_rows,num_of_columns = temp_matrix.shape
        plt.figure()
        tb = plt.table(cellText=np.round(temp_matrix,2), loc=(0,0), cellLoc='center')
        for cell in tb.properties()['child_artists']:
            cell.set_height(1/number_of_rows)
            cell.set_width(1/num_of_columns)

        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        plt.show()

class Hebbian(object):
    def __init__(self, input_dimensions=2,number_of_classes=4,transfer_function="Hard_limit",seed=None):
        """
        Initialize Perceptron model
        :param input_dimensions: The number of features of the input data, for example (height, weight) would be two features.
        :param number_of_classes: The number of classes.
        :param transfer_function: Transfer function for each neuron. Possible values are:
        "Hard_limit" ,  "Sigmoid", "Linear".
        :param seed: Random number generator seed.
        """
        if seed != None:
            np.random.seed(seed)
        self.input_dimensions = input_dimensions
        self.number_of_classes=number_of_classes
        self.transfer_function=transfer_function
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize the weights, initalize using random numbers.
        Note that number of neurons in the model is equal to the number of classes
        """
        self.weights =np.random.randn(self.number_of_classes,self.input_dimensions+1)


    def initialize_all_weights_to_zeros(self):
        """
        Initialize the weights, initalize using random numbers.
        """
        self.weights = np.zeros((self.number_of_classes,self.input_dimensions+1))


    def predict(self, X):
        """
        Make a prediction on an array of inputs
        :param X: Array of input [input_dimensions,n_samples]. Note that the input X does not include a row of ones
        as the first row.
        :return: Array of model outputs [number_of_classes ,n_samples]. This array is a numerical array.
        """
        n,m=X.shape
        X0 = np.ones((1,m))
        Xnew=np.vstack((X0,X))
        Y=np.dot(self.weights,Xnew)
        if self.transfer_function=="Hard_limit":
            output=np.where(Y<=0,0,1)
        elif self.transfer_function=="Sigmoid":
            output=1.0 / (1 + np.exp(-Y))
        elif self.transfer_function=="Linear":
            output=Y
        return output


    def print_weights(self):
        """
        This function prints the weight matrix (Bias is included in the weight matrix).
        """
        print(self.weights)

    "add ones to the input samples"
    def add_ones(self, T):      
        one = np.ones((1,T.shape[1]))
        final = np.vstack([one, T])
        return final

    def train(self, X, y, batch_size=1,num_epochs=10,  alpha=0.1,gamma=0.9,learning="Delta"):
        """
        Given a batch of data, and the necessary hyperparameters,
        this function adjusts the self.weights using Perceptron learning rule.
        Training should be repeted num_epochs time.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :param batch_size: number of samples in a batch
        :param num_epochs: Number of times training should be repeated over all input data
        :param alpha: Learning rate
        :param gamma: Controls the decay
        :param learning: Learning rule. Possible methods are: "Filtered", "Delta", "Unsupervised_hebb"
        :return: None
        """
        import math
        l = []
        flag = 0
        div = X.shape[1]//batch_size

        with_ones = self.add_ones(X)  

        one_hot = np.zeros((y.shape[0],self.number_of_classes))
        one_hot[np.arange(y.shape[0]),y] = 1
        one_hot = one_hot.transpose()


        if(with_ones.shape[1] % batch_size != 0):
            flag = 1

        axis = 1
        l = [batch_size*(i+1) for i  in range(math.floor(X.shape[axis] / batch_size))]  
        
        s = np.split(with_ones, l, axis = 1)
        t = np.split(one_hot,l, axis = 1)
        
        for i in range(0,num_epochs):
            for j in range(len(s)):
                delta_1 = np.dot(self.weights, s[j])
                fil = np.dot(t[j],s[j].transpose())

                if self.transfer_function == "Hard_limit":
                    main_op = np.where(delta_1<=0, 0, 1)   
                if(self.transfer_function == "Linear"):
                    main_op = delta_1
                if(self.transfer_function == "Sigmoid"):
                    main_op = 1.0/(1 + np.exp(-delta_1))
                
                
                if learning=="Delta":
                    error = np.subtract(t[j], main_op)
                    error = alpha*error
                    final = np.dot(error, s[j].transpose())
                    self.weights = self.weights + final
                elif learning=="Filtered":
                    self.weights = (1-gamma)*self.weights + alpha*fil
                elif learning=="Unsupervised_hebb":
                    un = alpha*main_op
                    self.weights = self.weights + np.dot(un,s[j].transpose())
    #return None 




    def calculate_percent_error(self,X, y):
        decoded = []
        from sklearn.metrics import accuracy_score

        with_ones = self.add_ones(X)

        y_pred = np.dot(self.weights, with_ones)

        if self.transfer_function == "Hard_limit":
            main_op = np.where(y_pred<=0, 0, 1)  
        if(self.transfer_function == "Linear"):
            main_op = y_pred
        if(self.transfer_function == "Sigmoid"):
            main_op = 1.0/(1 + np.exp(-y_pred))
            

        for i in range(main_op.shape[1]):
            decoded.append(np.argmax(main_op[:,i]))
            
        percent_error=round(1 - accuracy_score(list(y), decoded),2)

        return percent_error

        """
        Given a batch of data this function calculates percent error.
        For each input sample, if the predicted class output is not the same as the desired class,
        then it is considered one error. Percent error is number_of_errors/ number_of_samples.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return percent_error
        """


    def calculate_confusion_matrix(self,X,y):
        decoded = []
        from sklearn.metrics import confusion_matrix

        with_ones = self.add_ones(X)  

        y_pred = np.dot(self.weights, with_ones)

        if self.transfer_function == "Hard_limit":
            main_op = np.where(y_pred<=0, 0, 1)   
        if(self.transfer_function == "Linear"):
            main_op = y_pred
        if(self.transfer_function == "Sigmoid"):
            main_op = 1.0/(1 + np.exp(-y_pred))
            


        for i in range(main_op.shape[1]):
            decoded.append(np.argmax(main_op[:,i]))

        return(confusion_matrix(list(y), decoded))



        """
        Given a desired (true) output as one hot and the predicted output as one-hot,
        this method calculates the confusion matrix.
        If the predicted class output is not the same as the desired output,
        then it is considered one error.
        :param X: Array of input [input_dimensions,n_samples]
        :param y: Array of desired (target) outputs [n_samples]. This array includes the indexes of
        the desired (true) class.
        :return confusion_matrix[number_of_classes,number_of_classes].
        Confusion matrix should be shown as the number of times that
        an image of class n is classified as class m where 1<=n,m<=number_of_classes.
        """



if __name__ == "__main__":

    # Read mnist data
#    number_of_classes = 10
#    number_of_training_samples_to_use = 700
#    number_of_test_samples_to_use = 100
#    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
#    X_train_vectorized=((X_train.reshape(X_train.shape[0],-1)).T)[:,0:number_of_training_samples_to_use]
#    y_train = y_train[0:number_of_training_samples_to_use]
#    X_test_vectorized=((X_test.reshape(X_test.shape[0],-1)).T)[:,0:number_of_test_samples_to_use]
#    y_test = y_test[0:number_of_test_samples_to_use]
#    number_of_images_to_view=16
#    test_x=X_train_vectorized[:,0:number_of_images_to_view].T.reshape((number_of_images_to_view,28,28))
#    display_images(test_x)
#    input_dimensions=X_test_vectorized.shape[0]
#    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
#                    transfer_function="Hard_limit",seed=5)
#    # model.initialize_all_weights_to_zeros()
#    percent_error=[]
#    for k in range (10):
#        model.train(X_train_vectorized, y_train,batch_size=300, num_epochs=2, alpha=0.1,gamma=0.1,learning="Delta")
#        percent_error.append(model.calculate_percent_error(X_test_vectorized,y_test))
#    print("******  Percent Error ******\n",percent_error)
#    confusion_matrix=model.calculate_confusion_matrix(X_test_vectorized,y_test)
#    print(np.array2string(confusion_matrix, separator=","))


################################################################################################################################

    number_of_classes = 10
    number_of_training_samples_to_use = 1000
    number_of_test_samples_to_use = 100
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train_vectorized = ((X_train.reshape(X_train.shape[0], -1)).T)[:, 0:number_of_training_samples_to_use]
    y_train = y_train[0:number_of_training_samples_to_use]
    X_test_vectorized = ((X_test.reshape(X_test.shape[0], -1)).T)[:, 0:number_of_test_samples_to_use]
    y_test = y_test[0:number_of_test_samples_to_use]
    input_dimensions = X_test_vectorized.shape[0]
    model = Hebbian(input_dimensions=input_dimensions, number_of_classes=number_of_classes,
                    transfer_function="Hard_limit", seed=5)
    model.initialize_all_weights_to_zeros()
    percent_error = []
    for k in range(10):
        model.train(X_train_vectorized, y_train, batch_size=300, num_epochs=2, alpha=0.1, gamma=0.1, learning="Delta")
        percent_error.append(model.calculate_percent_error(X_test_vectorized, y_test))
    print(percent_error)
    confusion_matrix = model.calculate_confusion_matrix(X_test_vectorized, y_test)
    print(confusion_matrix)
    assert (np.array_equal(confusion_matrix, np.array( \
        [[8., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 13., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 7., 0., 0., 0., 0., 0., 0., 0.],
         [2., 0., 1., 8., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 1., 12., 0., 0., 0., 0., 0.],
         [4., 0., 1., 0., 0., 2., 0., 0., 0., 0.],
         [3., 0., 2., 0., 0., 0., 5., 0., 0., 0.],
         [1., 0., 0., 2., 0., 0., 0., 11., 0., 1.],
         [2., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
         [1., 0., 0., 0., 0., 0., 0., 1., 0., 9.]]))) or \
           (np.array_equal(confusion_matrix, np.array( \
               [[8., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 13., 0., 0., 0., 0., 0., 0., 0., 0.],
                [1., 0., 6., 0., 0., 0., 1., 0., 0., 0.],
                [2., 0., 1., 8., 0., 0., 0., 0., 0., 0.],
                [2., 0., 0., 1., 11., 0., 0., 0., 0., 0.],
                [4., 0., 1., 0., 0., 2., 0., 0., 0., 0.],
                [4., 0., 1., 0., 0., 0., 5., 0., 0., 0.],
                [2., 0., 0., 1., 0., 0., 0., 12., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
                [3., 0., 0., 0., 0., 0., 0., 3., 0., 5.]])))

    assert np.allclose(percent_error,
    np.array([0.74, 0.35, 0.32, 0.3, 0.28, 0.32, 0.25, 0.26, 0.3, 0.25]),rtol=1e-3, atol=1e-3) or \
           np.allclose(percent_error,
                        np.array([0.8 ,0.37,0.36,0.32,0.31,0.31,0.29,0.29,0.24,0.29]), rtol=1e-3, atol=1e-3)