# Standard python numerical analysis imports:

import numpy as np
#from scipy.signal import windows
#from scipy.interpolate import interp1d
#from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
#import h5py
#import json


from scipy.io import wavfile


# function to whiten data
def whiten(strain, interp_psd, dt):
    Nt = len(strain)
    freqs = np.fft.rfftfreq(Nt, dt)
    freqs1 = np.linspace(0, 2048, Nt // 2 + 1)

    # whitening: transform to freq domain, divide by asd, then transform back, 
    # taking care to get normalization right.
    hf = np.fft.rfft(strain)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht




# make wav (sound) files from the whitened data, +-2s around the event.


# function to keep the data within integer limits, and write to wavfile:
def write_wavfile(filename,fs,data):
    d = np.int16(data/np.max(np.abs(data)) * 32767 * 0.9)
    wavfile.write(filename,int(fs), d)



# function that shifts frequency of a band-passed signal
def reqshift(data,fshift=100,sample_rate=4096):
    """Frequency shift the signal by constant
    """
    x = np.fft.rfft(data)
    T = len(data)/float(sample_rate)
    df = 1.0/T
    nbins = int(fshift/df)
    # print T,df,nbins,x.real.shape
    y = np.roll(x.real,nbins) + 1j*np.roll(x.imag,nbins)
    y[0:nbins]=0.
    z = np.fft.irfft(y)
    return z




import matplotlib.pyplot as plt
import numpy as np
import os

def make_detector_plots(det, strain_L1_whitenbp, strain_H1_whitenbp, template_match, 
                        time, timemax, SNR, tevent, eventname, plottype,
                        template_fft, datafreq, d_eff, freqs, data_psd, fs):
    """
    Generates and saves a series of plots for matched filter results.
    
    This function creates:
    1. SNR time series plot (full and zoomed).
    2. Whitened data + template overlay plot.
    3. Residual plot.
    4. ASD and template frequency plot.
    
    It also returns the correctly copied template for use in the main script.
    """
    
    # Ensure the output directories exist
    os.makedirs('figures', exist_ok=True)
    os.makedirs('data', exist_ok=True) # The ASD plot saves to 'data/'

    # plotting changes for the detectors:
    if det == 'L1': 
        pcolor = 'g'
        strain_whitenbp = strain_L1_whitenbp
        template_out = template_match.copy()
    else:
        pcolor = 'r'
        strain_whitenbp = strain_H1_whitenbp
        template_out = template_match.copy()

    # -- Plot the result (SNR time series)
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-timemax, SNR, pcolor,label=det+' SNR(t)')
    #plt.ylim([0,25.])
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.legend(loc='upper left')
    plt.title(det+' matched filter SNR around event')

    # zoom in
    plt.subplot(2,1,2)
    plt.plot(time-timemax, SNR, pcolor,label=det+' SNR(t)')
    plt.grid('on')
    plt.ylabel('SNR')
    plt.xlim([-0.15,0.05])
    #plt.xlim([-0.3,+0.3])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax))
    plt.legend(loc='upper left')
    plt.savefig('figures/'+eventname+"_"+det+"_SNR."+plottype)
    plt.close() # Close the figure to save memory

    # -- Plot data vs template
    plt.figure(figsize=(10,8))
    plt.subplot(2,1,1)
    plt.plot(time-tevent,strain_whitenbp,pcolor,label=det+' whitened h(t)')
    plt.plot(time-tevent,template_match,'k',label='Template(t)')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax)) # User's code has timemax here
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det+' whitened data around event')

    # -- Plot residual
    plt.subplot(2,1,2)
    plt.plot(time-tevent,strain_whitenbp-template_match,pcolor,label=det+' resid')
    plt.ylim([-10,10])
    plt.xlim([-0.15,0.05])
    plt.grid('on')
    plt.xlabel('Time since {0:.4f}'.format(timemax)) # User's code has timemax here
    plt.ylabel('whitened strain (units of noise stdev)')
    plt.legend(loc='upper left')
    plt.title(det+' Residual whitened data after subtracting template around event')
    plt.savefig('figures/'+eventname+"_"+det+"_matchtime."+plottype)
    plt.close() # Close the figure

    # -- Display PSD and template
    # must multiply by sqrt(f) to plot template fft on top of ASD:
    plt.figure(figsize=(10,6))
    
    # Calculate template in frequency domain for plot
    # We use np.abs(datafreq) to avoid issues with negative frequencies
    template_f = np.absolute(template_fft)*np.sqrt(np.abs(datafreq)) / d_eff
    
    plt.loglog(datafreq, template_f, 'k', label='template(f)*sqrt(f)')
    plt.loglog(freqs, np.sqrt(data_psd),pcolor, label=det+' ASD')
    plt.xlim(20, fs/2)
    plt.ylim(1e-24, 1e-20)
    plt.grid()
    plt.xlabel('frequency (Hz)')
    plt.ylabel('strain noise ASD (strain/rtHz), template h(f)*rt(f)')
    plt.legend(loc='upper left')
    plt.title(det+' ASD and template around event')
    plt.savefig('data/'+eventname+"_"+det+"_matchfreq."+plottype)
    plt.close() # Close the figure

    # Return the copied template
    return template_out
