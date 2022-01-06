from scipy.fftpack.helper import next_fast_len
from obspy.signal.filter import bandpass
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import pyasdf
import obspy
import scipy
import glob
import os

'''
this script finds all station-pairs within a small region and use the cross-correlation functions 
for all these pairs resulted from 1-year daily stacking of Kanto data to perform integral transformation.
it then extracts the dispersion info from the local minimum

by Chengxin Jiang (chengxin_jiang@fas.harvard.edu) @ Aug/2019
'''

#######################################
########## PARAMETER SECTION ##########
#######################################

# absolute path for stacked data
data_path = '/Volumes/Seagate/research_Harvard/Kanto_basin/stacked'

# loop through each station
sta_file = 'station.lst'
locs = pd.read_csv(sta_file)
net  = locs['network']
sta  = locs['station']
lon  = locs['longitude']
lat  = locs['latitude']

# different components
dtype = 'Allstack0linear'
ccomp  = ['ZR','ZT','ZZ','RR','RT','RZ','TR','TT','TZ']
pindx  = [1,0,1,1,0,1,0,2,0]
onelag = False
norm   = True
bdpass = True
dist_inc = 0.5
stack_method = dtype.split('0')[-1]

# basic parameters for loading data
freqmin  = 0.1
freqmax  = 1
maxdist  = 80
maxnpair = 50
lag      = 160

# loop through each source station
for ista in range(17,18):

    # find all stations within maxdist
    sta_list = []
    nsta_list = []
    for ii in range(len(sta)):
        dist,_,_ = obspy.geodetics.base.gps2dist_azimuth(lat[ista],lon[ista],lat[ii],lon[ii])
        if dist/1000 < maxdist:
            sta_list.append(net[ii]+'.'+sta[ii])
        if dist/1000 < maxdist+4:
            nsta_list.append(net[ii]+'.'+sta[ii])
    nsta = len(sta_list)
    print(sta_list)

    # construct station pairs from the found stations
    allfiles = []
    '''
    for ii in range(nsta-1):
        for jj in range(ii+1,nsta):
            tfile1 = data_path+'/'+sta_list[ii]+'/'+sta_list[ii]+'_'+sta_list[jj]+'.h5'
            tfile2 = data_path+'/'+sta_list[jj]+'/'+sta_list[jj]+'_'+sta_list[ii]+'.h5'
            if os.path.isfile(tfile1):
                allfiles.append(tfile1)
            elif os.path.isfile(tfile2):
                allfiles.append(tfile2)
    '''
    for ii in range(nsta):
        tfile1 = data_path+'/'+sta_list[ii]+'/'+sta_list[ii]+'_'+sta_list[ista]+'.h5'
        tfile2 = data_path+'/'+sta_list[ista]+'/'+sta_list[ista]+'_'+sta_list[ii]+'.h5'
        if os.path.isfile(tfile1):
            allfiles.append(tfile1)
        elif os.path.isfile(tfile2):
            allfiles.append(tfile2)
    nfiles = len(allfiles)
    print(nfiles)

    # give it another chance for larger region
    if nfiles<maxnpair:
        print('station %s has no enough pairs'%sta[ista])
        continue

    # extract common variables from ASDF file
    try:
        ds    = pyasdf.ASDFDataSet(allfiles[0],mode='r')
        dt    = ds.auxiliary_data[dtype][ccomp[0]].parameters['dt']
        maxlag= ds.auxiliary_data[dtype][ccomp[0]].parameters['maxlag']
    except Exception:
        #raise ValueError('cannot open %s to read'%allfiles[0])
        print('cannot open %s to read'%allfiles[0])
        continue

    plt.figure(figsize=(12,9))
    for path in ccomp:

        ####################################
        ########## LOAD CCFS DATA ##########
        ####################################

        # initialize array
        anpts = int(maxlag/dt)+1
        tnpts = int(lag/dt)+1
        tvec = np.arange(0,tnpts)*dt
        Nfft = int(next_fast_len(tnpts))
        dist = np.zeros(nfiles,dtype=np.float32)
        cc_array = np.zeros(shape=(nfiles,tnpts),dtype=np.float32)

        # loop through each cc file
        icc=0
        for ii in range(nfiles):
            tmp = allfiles[ii]

            # load the data into memory
            with pyasdf.ASDFDataSet(tmp,mode='r') as ds:
                try:
                    dist[icc] = ds.auxiliary_data[dtype][path].parameters['dist']
                    tdata    = ds.auxiliary_data[dtype][path].data[:]
                    data = tdata[anpts-1:anpts+tnpts-1]*0.5+np.flip(tdata[anpts-tnpts:anpts],axis=0)*0.5
                except Exception:
                    print("continue! cannot read %s "%tmp)
                    continue  
            
            cc_array[icc] = data
            if bdpass:
                cc_array[icc] = bandpass(data,freqmin,freqmax,int(1/dt),corners=4, zerophase=True)
            if norm:
                cc_array[icc] = 0.8*cc_array[icc]/max(np.abs(cc_array[icc]))
            icc+=1

        # remove bad ones
        cc_array = cc_array[:icc]
        dist     = dist[:icc]
        nfiles1   = icc

        #####################################
        ############ PLOTTING ###############
        #####################################

        #---plot the 2D dispersion image-----
        tmpt = '33'+str(ccomp.index(path)+1)
        plt.subplot(tmpt)

        # plot the move-out plots as matrix
        ndata  = np.zeros(shape=(nfiles1,tnpts),dtype=np.float32)
        ndist  = np.zeros(nfiles1,dtype=np.float32)

        # averaging cross correlation in a certain distance range
        for ic in range(nfiles1-1):
            tindx = np.where((dist[ic]>=ic*dist_inc)&(dist<(ic+1)*dist_inc))[0]
            if len(tindx):
                ndata[ic] = np.mean(cc_array[tindx],axis=0)
                ndist[ic] = (ic+0.5)*dist_inc

        # normalize waveforms 
        indx  = np.where(ndist>0)[0]
        ndata = ndata[indx]
        ndist = ndist[indx]
        for ii in range(ndata.shape[0]):
            ndata[ii] /= np.max(np.abs(ndata[ii]))

        # ready to plot figures
        plt.imshow(ndata,cmap='seismic',extent=[tvec[0],tvec[-1],ndist[0],ndist[-1]],aspect='auto',origin='lower')
        if ccomp.index(path)==1:
            plt.title('%s & %d pairs @%4.1f-%4.1f Hz' % (sta[ista],nfiles1,freqmin,freqmax))
        plt.xlabel('time [s]')
        plt.ylabel('distance [km]')

    # output figure to pdf files
    if not os.path.isdir(os.path.join(data_path,'figures')):
        os.mkdir(os.path.join(data_path,'figures'))
    outfname = data_path+'/figures/'+str(sta[ista])+'_'+stack_method+'_'+str(maxdist)+'km.pdf'
    print(outfname)
    plt.savefig(outfname, format='pdf', dpi=300)
    plt.close()
