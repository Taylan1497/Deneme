#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

class Network:
    
    def __init__(self):
        self.weights1=np.random.rand(250,16)  # embedding layer weights initialization, 250--input_size, 16-- output size
        self.weights2=np.random.rand(16,128)  # hidden layer with input size 16 and output size 128
        self.weights3=np.random.rand(128,250) # We have total of 250 words outputlayer should give this
        # no need bias for embedding layer
        self.bias2 = np.zeros(128) 
        self.bias3 = np.zeros(250)
    
    
    def forward_propagation(self,input_data,target_output):
        
        embedding = np.dot(input_data,self.weights1) # matrix multiplication of input_data and embedding layer weights
        # after embedding we need to multiply with hidden layer weights without any activation functions,
        hidden1 = np.dot(embedding,self.weights2)+self.bias2 
        hidden1_output = self.sigmoid(hidden1) # hidden layer activated output
        output_layer = np.dot(hidden1_output,self.weights3) + self.bias3 # output layer shoul take hidden1 output as an input and activate with softmax function
        softmax_output = self.softmax(output_layer) # propabilistic output
        
        # and now we should calculate loss after our forward propagation.
        
        loss = self.cross_entropy(softmax_output,target_output)
        
        return loss, softmax_output, embedding, hidden1_output # we return output of each layer.
    
    
    def backward_propagation(self,input_data,embedding,hidden1_output,softmax_output,target_output):
        d_error3 = softmax_output - target_output # d_error3: from loss to output layer
        #structure of weights gradient for each layer -- np.dot(input.T,output_error(d_error3))
        #structure of bias gradient for each layer -- np.sum(d_error3)
        weights3_grad = np.dot(hidden1_output.T, d_error3) # weight 3 gradients
        bias3_grad = np.sum(d_error3, axis=0) # bias 3 gradients
        
        d_error2 = np.dot(d_error3, self.weights3.T )* hidden1_output * (1 - hidden1_output) # d_error2: from loss to hidden layer
        ##
        weights2_grad = np.dot(embedding.T,d_error2) # weights2 gradient
        bias2_grad = np.sum(d_error2,axis=0)
        ###
        d_error1 = np.dot(d_error2, self.weights2.T) # from loss to embeddings
        weights1_grad = np.dot(input_data.T,d_error1)
                          
        return weights1_grad,weights2_grad,weights3_grad,bias2_grad,bias3_grad
    
    #parameters should be updated after we obtain gradients
    def update_params(self,d_weights1, d_weights2, d_weights3, d_bias2, d_bias3, learning_rate):
        self.weights1 = self.weights1 - learning_rate * d_weights1
        self.weights2 = self.weights2 - learning_rate * d_weights2
        self.weights3 = self.weights3 - learning_rate * d_weights3
        self.bias2 = self.bias2 - learning_rate * d_bias2
        self.bias3 = self.bias3 - learning_rate * d_bias3
        
        
    # Loss Functions, some loss functions are coded for just tries.
    # external links for the Runtime Error
    # https://stackoverflow.com/questions/38125319/python-divide-by-zero-encountered-in-log-logistic-regression
    # https://stackoverflow.com/questions/46510929/mle-log-likelihood-for-logistic-regression-gives-divide-by-zero-error
    # https://www.appsloveworld.com/machine-learning/34/mle-log-likelihood-for-logistic-regression-gives-divide-by-zero-error
    # log(p+epsilon) and log(1-p+epsilon) with a small positive epsilon value. This ensures that log(0.0) never happens.
    def cross_entropy(self,y_true,y_pred):
        loss = -np.sum(y_true * np.log(y_pred + 1e-10))/ len(y_true)
        return loss
    
   
    # 1e-10 term added to get rid of "RuntimeWarning: divide by zero encountered in log"
    def mse(self, y_true, y_pred):            
        loss = np.mean(np.power(y_true-y_pred, 2))
        return loss
    
    # Activation Functions
    
    def sigmoid(self, x):
        activated = 1/(1 + np.exp(-x))
        return activated
    
    #def softmax(self,x):
        #activated = np.exp(x)/np.sum(np.exp(x),axis=0)
        #return activated
    
        
    def softmax(self, x):
        # Solved "RuntimeWarning: overflow encountered in exp.
        x = x - x.max(1).reshape((-1, 1))
        e = np.exp(x)
        return e/e.sum(1).reshape((-1, 1))
    
    # extra activation function here.
    def tanh(self,x):
        activated = np.tanh(x)
        return activated
    
    #def calculate_accuracy(self,target_output,y_pred):
        #acc= 100 - np.mean(np.abs(target_output - y_pred))*100
        #print("train accuracy: {} %".format(100 - np.mean(np.abs(target_output - y_pred)) * 100))
        #return acc
    
    def calculate_accuracy2(self, y_true, y_pred):
        true_predicted = 0
        for i in range(len(y_true)):
            true_index = list(y_true[i]).index(1.)
            predicted_index = list(y_pred[i]).index(np.max(y_pred[i]))
            if true_index == predicted_index: 
                true_predicted += 1
        accuracy = 100. * true_predicted / (len(y_true))
        return accuracy


# In[ ]:




