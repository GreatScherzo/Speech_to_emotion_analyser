#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyaudio
import socket
import sys
import wave
from sklearn.externals import joblib

from importlib.machinery import SourceFileLoader
foo = SourceFileLoader("STD", r"C:\Users\Barzarin\OneDrive\jupyter\speech_to_data.py").load_module()
audio = pyaudio.PyAudio()


# In[2]:


info = audio.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
        if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            print ("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))


# In[3]:


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "file1.wav"
CLASSIFIER_PATH = r"C:\Users\Barzarin\OneDrive\jupyter\speech_emo_recognition2.pkl" 

# start Recording
stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

print("recording")

frames = []
 
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("finished recording")

#stop Recording
stream.stop_stream()
stream.close()
audio.terminate()
 
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

#classifying

classifier = joblib.load(CLASSIFIER_PATH)

result = foo.get_data(WAVE_OUTPUT_FILENAME)

#anger(W)=0, boredom(L)=1, disgust(E)=2, anxiety/fear(A)=3, happiness(F)=4, sad(T)=5, neutral(N)=6 

classifier.predict_proba(result)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




