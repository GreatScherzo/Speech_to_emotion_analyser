
# coding: utf-8

# In[ ]:

from __future__ import division
from numpy import polyfit, arange
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn 
from IPython.display import display
import scipy.io.wavfile as wav
import os
import speechpy as sp
import sys
get_ipython().run_line_magic('matplotlib', 'notebook')

import speechpy
import librosa as lb
import re # for regular expression
import statistics
#np.set_printoptions(threshold=sys.maxsize)
import scipy.stats as stats
from Signal_Analysis.features import signal as SA
np.set_printoptions(threshold=1000)


# In[ ]:

from numpy.fft import rfft
from numpy import argmax, mean, diff, log
from matplotlib.mlab import find
from scipy.signal import blackmanharris, fftconvolve
from time import time
import sys

from importlib.machinery import SourceFileLoader

def freq_from_autocorr(sig, fs):
    """
    Estimate frequency using autocorrelation
    """
    # Calculate autocorrelation (same thing as convolution, but with
    # one input reversed in time), and throw away the negative lags
    corr = fftconvolve(sig, sig[::-1], mode='full')
    corr = corr[len(corr)//2:]

    # Find the first low point
    d = diff(corr)
    start = find(d > 0)[0]

    # Find the next peak after the low point (other than 0 lag).  This bit is
    # not reliable for long signals, due to the desired peak occurring between
    # samples, and other peaks appearing higher.
    # Should use a weighting function to de-emphasize the peaks at longer lags.
    peak = argmax(corr[start:]) + start
    px, py = parabolic(corr, peak)

    return fs / px

def parabolic(f, x):
    """Quadratic interpolation for estimating the true position of an
    inter-sample maximum when nearby samples are known.
   
    f is a vector and x is an index for that vector.
   
    Returns (vx, vy), the coordinates of the vertex of a parabola that goes
    through point x and its two neighbors.
   
    Example:
    Defining a vector f with a local maximum at index 3 (= 6), find local
    maximum if points 2, 3, and 4 actually defined a parabola.
   
    In [3]: f = [2, 3, 1, 6, 4, 2, 3, 1]
   
    In [4]: parabolic(f, argmax(f))
    Out[4]: (3.2142857142857144, 6.1607142857142856)
   
    """
    xv = 1/2. * (f[x-1] - f[x+1]) / (f[x-1] - 2 * f[x] + f[x+1]) + x
    yv = f[x] - 1/4. * (f[x-1] - f[x+1]) * (xv - x)
    return (xv, yv)


def parabolic_polyfit(f, x, n):
    """Use the built-in polyfit() function to find the peak of a parabola
    
    f is a vector and x is an index for that vector.
    
    n is the number of samples of the curve used to fit the parabola.
    """    
    a, b, c = polyfit(arange(x-n//2, x+n//2+1), f[x-n//2:x+n//2+1], 2)
    xv = -0.5 * b/a
    yv = a * xv**2 + b * xv + c
    return (xv, yv)


if __name__=="__main__":
    from numpy import argmax
    import matplotlib.pyplot as plt
    
    y = [2, 1, 4, 8, 11, 10, 7, 3, 1, 1]
    
    xm, ym = argmax(y), y[argmax(y)]
    xp, yp = parabolic(y, argmax(y))
    
    plot = plt.plot(y)
    plt.hold(True)
    plt.plot(xm, ym, 'o', color='silver')
    plt.plot(xp, yp, 'o', color='blue')
    plt.title('silver = max, blue = estimated max')


# In[ ]:


mfcc_data = []
mfccstd = []
mfccskew = []
mfcckurt = []
mfccmax = []
mfccmin = []

zcrmean = []
zcrstd = []
zcrskew = []
zcrkurt = []
zcrmax = []
zcrmin = []

#f0mean = []
#f0std = []
#f0skew = []
#f0kurt = []
#f0max = []
#f0min = []

rmsmean = []
rmsstd = []
rmsskew = []
rmskurt = []
rmsmax = []
rmsmin = []

hnrmean = []
hnrstd = []
hnrskew = []
hnrkurt = []
hnrmax = []
hnrmin = []


def get_data(inputwave):
    fs, signal = wav.read(inputwave)
    y, sr = lb.load(inputwave)
    
    
    
    mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
                                num_filters=40, fft_length=512, low_frequency=0, high_frequency=None, num_cepstral=12)
    
    
    zcr = lb.feature.zero_crossing_rate(y)
    
    f0 = freq_from_autocorr(signal, fs)
    
    rms = lb.feature.rmse(y=y)
    
    
    HNR = SA.get_HNR(signal, fs, time_step=0, min_pitch=75, 
                                  silence_threshold=0.1, periods_per_window=4.5)
        
        
        
        
        
        
        
################################################################################################
        
      
    print(inputwave)
    
    mfcc_data.append(mfcc.mean(axis=0))
    mfccstd.append(np.std(mfcc, axis=0))
    mfccskew.append(stats.skew(mfcc, axis=0))
    mfcckurt.append(stats.kurtosis(mfcc, axis=0))
    mfccmax.append(np.max(mfcc, axis=0))
    mfccmin.append(np.min(mfcc, axis=0))
    
    
    
    zcrmean.append(zcr.mean(axis=1))
    zcrstd.append(np.std(zcr, axis=1))
    zcrskew.append(stats.skew(zcr, axis=1))
    zcrkurt.append(stats.kurtosis(zcr, axis=1))
    zcrmax.append(np.max(zcr, axis=1))
    zcrmin.append(np.min(zcr, axis=1))
    
    
    #f0mean.append(f0.mean(axis=0))
    #f0std.append(f0.std(axis=0))
    #f0skew.append(stats.skew(f0, axis=0))
    #f0kurt.append(stats.kurtosis(f0, axis=0))
    #f0max.append(np.max(f0, axis=0))
    #f0min.append(np.min(f0, axis=0))
    
    
    rmsmean.append(rms.mean(axis=1))
    rmsstd.append(np.std(rms, axis=1))
    rmsskew.append(stats.skew(rms, axis=1))
    rmskurt.append(stats.kurtosis(rms, axis=1))
    rmsmax.append(np.max(rms, axis=1))
    rmsmin.append(np.min(rms, axis=1))
    
    hnrmean.append(HNR.mean(axis=0))
    hnrstd.append(HNR.std(axis=0))
    hnrskew.append(stats.skew(HNR, axis=0))
    hnrkurt.append(stats.kurtosis(HNR, axis=0))
    hnrmax.append(np.max(HNR, axis=0))
    hnrmin.append(np.min(HNR, axis=0))
#######################################################################################################

    mfcc_array = np.asarray(mfcc_data)
    mfccstd_array = np.asarray(mfccstd)
    mfccskew_array = np.asarray(mfccskew)
    mfcckurt_array = np.asarray(mfcckurt)
    mfccmax_array = np.asarray(mfccmax)
    mfccmin_array = np.asarray(mfccmin)
    
    zcrmean_array = np.asarray(zcrmean)
    zcrstd_array = np.asarray(zcrstd)
    zcrskew_array = np.asarray(zcrskew)
    zcrkurt_array = np.asarray(zcrkurt)
    zcrmax_array = np.asarray(zcrmax)
    zcrmin_array = np.asarray(zcrmin)
    
    
    rmsmean_array = np.asarray(rmsmean)
    rmsstd_array = np.asarray(rmsstd)
    rmsskew_array = np.asarray(rmsskew)
    rmskurt_array = np.asarray(rmskurt)
    rmsmax_array = np.asarray(rmsmax)
    rmsmin_array = np.asarray(rmsmin)
    
    hnrmean_array = np.asarray(hnrmean)
    hnrstd_array = np.asarray(hnrstd)
    hnrskew_array = np.asarray(hnrskew)
    hnrkurt_array = np.asarray(hnrkurt)
    hnrmax_array = np.asarray(hnrmax)
    hnrmin_array = np.asarray(hnrmin)
    
    hnrmean_rearray=np.reshape(hnrmean_array, (-1,1))
    hnrstd_rearray=np.reshape(hnrstd_array, (-1,1))
    hnrskew_rearray=np.reshape(hnrskew_array, (-1,1))
    hnrkurt_rearray=np.reshape(hnrkurt_array, (-1,1))
    hnrmax_rearray=np.reshape(hnrmax_array, (-1,1))
    hnrmin_rearray=np.reshape(hnrmin_array, (-1,1))



    
    finaldata = np.concatenate((mfcc_array, mfccstd_array,
                            mfccskew_array, mfcckurt_array,
                            mfccmax_array, mfccmin_array,
                            zcrmean_array, zcrstd_array,
                            zcrskew_array, zcrkurt_array,
                            zcrmax_array, zcrmin_array,
                            rmsmean_array, rmsstd_array,
                            rmsskew_array, rmskurt_array,
                            rmsmax_array, rmsmin_array,
                            hnrmean_rearray, hnrstd_rearray,
                            hnrskew_rearray, hnrkurt_rearray,
                            hnrmax_rearray, hnrmin_rearray
                           ), axis = 1)
    
    return finaldata
        

