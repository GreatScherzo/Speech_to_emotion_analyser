#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyaudio
import socket
import sys
import wave
from sklearn.externals import joblib
import os
import numpy as np
np.set_printoptions(threshold=np.inf, precision=3)
from sklearn.preprocessing import StandardScaler

from importlib.machinery import SourceFileLoader
foo = SourceFileLoader("STD", r"get_data_final.py").load_module()
audio = pyaudio.PyAudio()


# In[2]:


DIR = r"Speech_emotion_filtering"
CLASSIFIER_PATH = r"speech_emo_recognitionkaizenscaler.pkl" 
classifier = joblib.load(CLASSIFIER_PATH)


# In[3]:


for inputwave in os.listdir(DIR):
    np.set_printoptions(threshold=np.inf)
    
    if inputwave.endswith(".wav"):

        #anger(W)=0, boredom(L)=1, disgust(E)=2, anxiety/fear(A)=3, happiness(F)=4, sad(T)=5, neutral(N)=6 
        #classifier.predict_proba(result)
        
        result = foo.get_data(inputwave)
        
        Result_prob = classifier.predict_proba(StandardScaler().fit_transform(result))
        
        if inputwave.endswith("201Label.wav"):
            with open('policescaler.txt', 'a+') as outfile:
                outfile.write(str(Result_prob))


# In[ ]:


print(Result_prob).shape


# In[ ]:


with open('resultfreakoutvid3.txt', 'a+') as outfile:
    outfile.write(str(Result_prob)+',\n')


# In[ ]:




