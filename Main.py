#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from Network import Network
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

target_dir="./data/data/"
data_vocab=np.load(target_dir+"vocab.npy")
#batch_size = 100
#number_of_epoch = 20
#learning_rate = 0.02
#number_of_batch = int(train_input.shape[0]/batch_size)


def plotting(train_accuracy,validation_accuracy,epochs):
    accuracy_df=pd.DataFrame({"Train Accuracy":train_accuracy,"Validation Accuracy":validation_accuracy},
                              index=epochs)
    plot = accuracy_df.plot(xlabel="Epochs",ylabel="Train and Validation Accuracy(%)").get_figure()
    plot.savefig('Plots/Train_and_Validation.png')
    #return plot

def Load_Shuffle_Data(target_dir):
    train_input = np.load(target_dir+"train_inputs.npy")
    train_target = np.load(target_dir+"train_targets.npy").reshape(len(train_input), 1)
    train_data = np.concatenate((train_input, train_target), axis=1)
    np.random.shuffle(train_data)
    train_input, train_target = train_data[:, 0:3], train_data[:, 3]
    data_vocab=np.load(target_dir+"vocab.npy")

    
    val_input = np.load(target_dir+"valid_inputs.npy")
    val_target = np.load(target_dir+"valid_targets.npy")
    print("Train and Validation Inputs and Targets loaded, Train shuffled.")
    print("Train Input Shape:",train_input.shape,"Train Target Shape:",train_target.shape,"Val Shape:",
          val_input.shape,"Val Target",val_target.shape)
    return train_input,train_target,val_input,val_target,data_vocab

def one_hot_encode(train_input):
    one_hot_data_train=[]
    for i in train_input:
        array_element=[0]*len(data_vocab)
        for k in i:
            array_element[k]=1
        one_hot_data_train.append(array_element)
    return one_hot_data_train



acc_train = []
acc_valid =[]


def main():
    
    train_input,train_target,val_input,val_target,data_vocab = Load_Shuffle_Data(target_dir)
    #target_output = np.eye(250)[train_target[i*batch_size:(i+1)*batch_size]]
    train_input=np.array(one_hot_encode(train_input))
    val_input_hot = np.array(one_hot_encode(val_input))
    batch_size = 100
    number_of_epoch = 20
    learning_rate = 0.02
    number_of_batch = int(train_input.shape[0]/batch_size)
    
    #accuracies_train = []
    #accuracies_validation = []
    network = Network()
    #Start Training
    for epoch in range(number_of_epoch):
        print("Epoch: ", epoch)
        for i in range(number_of_batch):
            input_batch = np.array(train_input[i*batch_size:(i+1)*batch_size])
            #target_batch = train_target[i*batch_size:(i+1)*batch_size]
            target_batch = np.eye(250)[train_target[i*batch_size:(i+1)*batch_size]]
            loss_train, softmax_output, embedding, hidden1_output = network.forward_propagation(input_batch, target_batch)
        
            d_w1, d_w2, d_w3, d_b2, d_b3 = network.backward_propagation(input_batch,embedding, hidden1_output,softmax_output, target_batch)
            
            network.update_params(d_w1, d_w2, d_w3, d_b2, d_b3, learning_rate)
            
        
        # Calculate Train and Validation Accuracies after each epoch 
        expected_outputs=np.eye(250)[train_target]
        loss, probabilities, _, _ = network.forward_propagation(train_input,expected_outputs)
        accuracy = network.calculate_accuracy2(expected_outputs, probabilities)
        acc_train.append(accuracy)
        
        expected_val = np.eye(250)[val_target]
        loss_val,prob_val, _, _, = network.forward_propagation(val_input_hot,expected_val)
        acc_val = network.calculate_accuracy2(expected_val,prob_val)
        acc_valid.append(acc_val)
    
    #Plotting
    
    epochs = [i for i in range(number_of_epoch)]
    plotting(acc_train,acc_valid,epochs)
    
    #Print Last Train and Validation Accuracy
    print("Final train accuracy is {:.3f}".format(acc_train[len(epochs)-1]), "%")
    print("Final validation accuracy is {:.3f}".format(acc_valid[len(epochs)-1]), "%")
    
    #Save Model
    
    pickle.dump(network, open('new_model_Apr3.pk', 'wb'))
    
    print("Model is saved.")
    
main()    
    

