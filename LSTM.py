
# coding: utf-8

# In[1]:


import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import metrics
plt.style.use("ggplot")
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# In[2]:


s=[]
sentences = []
filepath = 'ner.txt'  
with open(filepath, 'r', encoding='latin1') as fp:
    for cnt, line in enumerate(fp):
        if(line!='\n'):
            s.append(tuple((line.strip()).split(" ")))
        else:
            sentences.append(s)
            s=[]
#         if(line!='\n'):
#             print(line.strip())


# In[3]:


words=[]
for sent in sentences:
    for t in sent:
        words.append(t[0])

words = list(set(words))
words.append('ENDPAD')
n_words = len(words)
tags = ['O', 'T', 'D']
n_tags = len(tags)


# In[4]:


plt.hist([len(s) for s in sentences], bins=50)
plt.show()


# In[5]:


max_len = max([len(x) for x in sentences])
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


# In[6]:


X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=n_words - 1)

y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])

y = [to_categorical(i, num_classes=n_tags) for i in y]


# In[7]:


X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)


# In[8]:


from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional


# In[9]:


input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words, output_dim=50, input_length=max_len)(input)
model = Dropout(0.1)(model)
model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
out = TimeDistributed(Dense(n_tags, activation="softmax"))(model)  # softmax output layer


# In[10]:


model = Model(input, out)
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])


# In[20]:


history = model.fit(X_tr, np.array(y_tr), batch_size=256, epochs=10, validation_data=(X_te, np.array(y_te)), verbose=1)


# In[21]:


scores = model.evaluate(X_te, np.array(y_te))
print('Test accuracy(%): ', scores[1])


# In[22]:


hist = pd.DataFrame(history.history)


# In[23]:


plt.figure(figsize=(18, 16), dpi= 200, facecolor='w', edgecolor='k')
plt.rcParams.update({'font.size': 22})
plt.plot(hist["acc"]*100, label = 'Training accuracy')
plt.plot(hist["val_acc"]*100, label = 'Validation accuracy')
plt.legend()
plt.xlabel('Number of Epochs')
plt.ylabel('% Accuracy')
plt.show()
#plt.savefig("NLU_Assignment3_LSTM")

