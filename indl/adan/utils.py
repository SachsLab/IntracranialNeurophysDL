import numpy as np 
import scipy.ndimage

def load_data(raw_data): 
    Bin = 20.
    raw_data = raw_data['binnedData']
    spike_data = raw_data['spikeratedata']/Bin
    spike_data_smooth = np.zeros_like(spike_data)
    for ii in range(spike_data.shape[1]):
        spike_data_smooth[:,ii] = scipy.ndimage.gaussian_filter1d(spike_data[:,ii],10/4)
    emg_data = raw_data['emgdatabin']*1.
    tinfo = raw_data['trialtable']*1.
    # removing unsuccessful trials
    rindex = np.where(tinfo[:,8] != ord('R'))
    tinfo = np.delete(tinfo,rindex,0)
    spike = np.zeros([0,np.shape(spike_data)[1]])
    emg = np.zeros([0,np.shape(emg_data)[1]])
    for ii in range(len(tinfo)):
        tidx = np.ceil(tinfo[ii,6:8]*Bin).astype(int)
        s = spike_data_smooth[tidx[0]:tidx[1],:]
        e = emg_data[tidx[0]:tidx[1],:]
        spike = np.append(spike,s,axis=0)
        emg  = np.append(emg,e,axis=0)
    # EMG outlier removal
    outlier = np.mean(emg,axis=0)+4*np.std(emg,axis=0)
    for ii in range(len(outlier)):
        emg[:,ii] = emg[:,ii].clip(max=outlier[ii])
    return spike,emg

def get_batches(x,batch_size):
    n_batches = len(x)//(batch_size)
    x = x[:n_batches*batch_size:]
    for n in range(0, x.shape[0],batch_size):
        x_batch = x[n:n+(batch_size)]
        yield x_batch

def vaf(x,xhat):
    x = x - x.mean(axis=0)
    xhat = xhat - xhat.mean(axis=0)
    return (1-(np.sum(np.square(x - xhat))/np.sum(np.square(x))))*100


