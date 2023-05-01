#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

vocab = np.load('data/data/vocab.npy')
print("vocab.npy is loaded.")
one_hot_matrix = np.identity(250)

network = pickle.load(open('new_model_Apr3.pk','rb'))
print("model is loaded.")

embeddings = np.dot(one_hot_matrix, network.weights1) # (250x16)
embeddings_2d = TSNE(n_components=2).fit_transform(embeddings) # (250x2)
print("2D embeddings created.")
np.set_printoptions(suppress=True)

x_coords, y_coords = embeddings_2d[:, 0], embeddings_2d[:, 1]

plt.figure(figsize=(10,6))
plt.scatter(x_coords, y_coords, s=0)

for label, x, y in zip(vocab, x_coords, y_coords):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')


plt.savefig('tsne.png')
print("tsne.png saved.")
plt.show()

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

