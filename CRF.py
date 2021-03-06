
# coding: utf-8

# In[65]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics


# In[66]:


s=[]
sents = []
filepath = 'ner.txt'  
with open(filepath, 'r') as fp:
    for cnt, line in enumerate(fp):
        if(line!='\n'):
            s.append(tuple((line.strip()).split(" ")))
        else:
            sents.append(s)
            s=[]
#         if(line!='\n'):
#             print(line.strip())


# In[67]:


train_sents, test_sents = train_test_split(sents, test_size=0.2, random_state=42)


# In[68]:


def word2features(sent, i):
    word = sent[i][0]
    #postag = sent[i][1]
    
    features = {
        'bias': 1.0,
        #'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word length': len(word)
#         'postag': postag,
#         'postag[:2]': postag[:2],        
    }
    if i > 0:
        word1 = sent[i-1][0]
        #postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1word length': len(word1)
#             '-1:postag': postag1,
#             '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        #postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1word length': len(word1)
#             '+1:postag': postag1,
#             '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]


# In[70]:


X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


# In[71]:


get_ipython().run_cell_magic('time', '', "crf = sklearn_crfsuite.CRF(\n    algorithm='lbfgs', \n    c1=0.1, \n    c2=0.1, \n    max_iterations=100, \n    all_possible_transitions=True\n)\ncrf.fit(X_train, y_train)")


# In[72]:


labels = list(crf.classes_)


# In[73]:


y_pred = crf.predict(X_test)
metrics.flat_f1_score(y_test, y_pred, 
                      average='weighted', labels=labels)


# In[74]:


# group B and I results
sorted_labels = sorted(
    labels, 
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, y_pred, labels=sorted_labels, digits=3
))


# In[75]:



def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(crf.state_features_).most_common(5))

print("\nTop negative:")
print_state_features(Counter(crf.state_features_).most_common()[-5:])


# In[76]:


from collections import Counter

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(crf.transition_features_).most_common(3))

print("\nTop unlikely transitions:")
print_transitions(Counter(crf.transition_features_).most_common()[-3:])

