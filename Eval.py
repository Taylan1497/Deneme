#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np

def one_hot_encode(train_input):
    one_hot_data_train=[]
    for i in train_input:
        array_element=[0]*len(vocab)
        for k in i:
            array_element[k]=1
        one_hot_data_train.append(array_element)
    return one_hot_data_train

def calculate_accuracy2(self, y_true, y_pred):
    true_predicted = 0
    for i in range(len(y_true)):
        true_index = list(y_true[i]).index(1.)
        predicted_index = list(y_pred[i]).index(np.max(y_pred[i]))
        if true_index == predicted_index: 
            true_predicted += 1
    accuracy = 100. * true_predicted / (len(y_true))
    return accuracy

print("Testing is started.")
test_input = np.load('data/data/test_inputs.npy')
test_target = np.load('data/data/test_targets.npy')
vocab = np.load('data/data/vocab.npy')

network = pickle.load(open('new_model_Apr3.pk','rb'))
print("model.pk is loaded.")
        
expected_test = np.eye(250)[test_target]
test_input_one = one_hot_encode(test_input)
loss_val,prob_test, _, _, = network.forward_propagation(test_input_one,expected_test)
acc_test = network.calculate_accuracy2(expected_test,prob_test)
#acc_valid.append(acc_val)


print("Test accuracy: {:.3f}".format(acc_test), "%")


# In[ ]:


## Predict Next Word
#network = pickle.load(open('new_model_Apr3.pk','rb'))
def predict_next_word(network,word1,word2,word3):
    vocab = np.load('data/data/vocab.npy')
    word_to_id = {}
    for i in range(len(vocab)):
        word_to_id[vocab[i]]=i
    #word_to_id
    #print(list(vocab).index(word1))
    try:
        word_input=np.zeros(250)

        word_input[list(vocab).index(word1)]=1
        word_input[list(vocab).index(word2)]=1
        word_input[list(vocab).index(word3)]=1
        word_input=word_input.reshape(1,250)

        empty_target=np.eye(250)[0]
    except:
        return "Word is not in the list."
    
    loss, probability, _, _ = network.forward_propagation(word_input, empty_target)

    index_of_next_word = list(probability[0]).index(np.max(probability[0]))
    
    return vocab[index_of_next_word]

print('city of new ->', predict_next_word(network, 'city', 'of', 'new'))
print('life in the ->', predict_next_word(network, 'life', 'in', 'the'))
print('he is the ->', predict_next_word(network, 'he', 'is', 'the'))

