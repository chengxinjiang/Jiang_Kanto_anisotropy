import os 
import sys
import glob
import pyasdf 
import numpy as np
import matplotlib.pyplot as plt 
from obspy.signal.filter import bandpass 

'''
display the moveout (2D matrix) of the cross-correlation functions stacked for all time chuncks.

INPUT parameters:
---------------------
sfile: cross-correlation functions outputed by S2
dtype: datatype either 'Allstack0pws' or 'Allstack0linear'
freqmin: min frequency to be filtered
freqmax: max frequency to be filtered
ccomp:   cross component
dist_inc: distance bins to stack over
disp_lag: lag times for displaying
savefig: set True to save the figures (in pdf format)
sdir: diresied directory to save the figure (if not provided, save to default dir)
'''

# data path (make sure rootpath is modified to your local ones)
rootpath = '/Users/chengxin/Documents/Kanto_basin/stacked'
source_list = ['E.SKMM']

# some input parameters
dtype  = 'Allstack0linear'
fmin   = 0.1
fmax   = 1
dist_inc = 1
disp_lag = 160
ccomp  = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']

# position matrix
post1 = [0,0,0,1,1,1,2,2,2]
post2 = [0,1,2,0,1,2,0,1,2]

# loop through each source
for source in source_list:
    sfiles = glob.glob(os.path.join(rootpath,'*/*'+source+'*.h5'))

    # extract common variables of the CCFs
    try:
        ds    = pyasdf.ASDFDataSet(sfiles[0],mode='r')
        dt    = ds.auxiliary_data[dtype][ccomp[2]].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][ccomp[2]].parameters['maxlag']
        stack_method = dtype.split('0')[-1]
    except Exception:
        raise ValueError("exit! cannot open %s to read"%sfiles[0])

    # lags for display   
    if not disp_lag:
        disp_lag=maxlag
    if disp_lag>maxlag:
        raise ValueError('lag excceds maxlag!')

    # construct time series
    t = np.arange(-int(disp_lag),int(disp_lag)+dt,step=(int(2*int(disp_lag)/4)))
    indx1 = int((maxlag-disp_lag)/dt)
    indx2 = indx1+2*int(disp_lag/dt)+1

    # cc matrix
    nwin = len(sfiles)
    data = np.zeros(shape=(nwin,indx2-indx1),dtype=np.float32)
    dist = np.zeros(nwin,dtype=np.float32)
    ngood= np.zeros(nwin,dtype=np.int16)    

    mdist = 80
    fig,ax = plt.subplots(3,3,figsize=(12,9), sharex=True)
    # loop through all 9 components
    for ic in range(len(ccomp)):
        comp = ccomp[ic]
        pos1 = post1[ic]
        pos2 = post2[ic]

        # load cc and parameter matrix
        for ii in range(len(sfiles)):
            sfile = sfiles[ii]
            iflip = 0
            treceiver = sfile.split('_')[-1]
            if treceiver == source+'.h5':
                iflip = 1

            ds = pyasdf.ASDFDataSet(sfile,mode='r')
            try:
                # load data to variables
                dist[ii] = ds.auxiliary_data[dtype][comp].parameters['dist']
                ngood[ii]= ds.auxiliary_data[dtype][comp].parameters['ngood']
                tdata    = ds.auxiliary_data[dtype][comp].data[indx1:indx2]
            except Exception:
                print("continue! cannot read %s "%sfile);continue

            # do move-out within certain distance
            if dist[ii] > mdist:continue

            if iflip:
                data[ii] = bandpass(np.flip(tdata,axis=0),fmin,fmax,int(1/dt),corners=4, zerophase=True)
            else:
                data[ii] = bandpass(tdata,fmin,fmax,int(1/dt),corners=4, zerophase=True)

        # average the CCFs at a distance bin
        ntrace = int(np.round(mdist)/dist_inc)
        ndata  = np.zeros(shape=(ntrace,indx2-indx1),dtype=np.float32)
        ndist  = np.zeros(ntrace,dtype=np.float32)
        for td in range(0,ntrace-1):
            tindx = np.where((dist>=td*dist_inc)&(dist<(td+1)*dist_inc))[0]
            if len(tindx):
                ndata[td] = np.mean(data[tindx],axis=0)
                ndist[td] = (td+0.5)*dist_inc

        # normalize waveforms for visulization
        indx  = np.where(ndist>0)[0]
        ndata = ndata[indx]
        ndist = ndist[indx]
        for ii in range(ndata.shape[0]):
            print(ii,np.max(np.abs(ndata[ii])))
            ndata[ii] /= np.max(np.abs(ndata[ii]))

        # plotting each subfigure
        ax[pos1,pos2].matshow(ndata,cmap='jet',extent=[-disp_lag,disp_lag,ndist[0],ndist[-1]],aspect='auto',origin='lower')
        if ic==0 or ic==3 or ic==6:
            ax[pos1,pos2].plot([0,160],[0,80],'r--',linewidth=1)
            ax[pos1,pos2].plot([0,80],[0,80],'g--',linewidth=1)
        if ic==1:
            ax[pos1,pos2].set_title('%s @%5.3f-%5.2f Hz'%(stack_method,fmin,fmax))
        if ic==6 or ic==7 or ic==8:
            ax[pos1,pos2].set_xlabel('time [s]')
        if ic==0 or ic==3 or ic==6:
            ax[pos1,pos2].set_ylabel('distance [km]')
        ax[pos1,pos2].set_xticks(t)
        ax[pos1,pos2].xaxis.set_ticks_position('bottom')
        ax[pos1,pos2].plot([0,0],[0,mdist],'b--',linewidth=1)
        font = {'family': 'serif', 'color':  'red', 'weight': 'bold','size': 16}
        ax[pos1,pos2].text(disp_lag*0.7,10,comp,fontdict=font)
    
    # output the plot
    if not os.path.isdir(os.path.join(rootpath,'figures')):
        os.mkdir(os.path.join(rootpath,'figures'))
    outfname = rootpath+'/figures/figure4_{0:s}_{1:4.2f}_{2:4.2f}Hz.pdf'.format(source,fmin,fmax)
    fig.tight_layout()
    plt.close()
    fig.savefig(outfname, format='pdf', dpi=400)
