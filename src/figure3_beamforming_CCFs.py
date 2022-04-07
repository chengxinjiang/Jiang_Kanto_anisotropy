import os
import scipy
import obspy
import pyasdf
import numpy as np
import pandas as pd
from numba import jit
import matplotlib.pyplot as plt 
from scipy.fftpack.helper import next_fast_len

'''
performs beamforming to estimate the ray slowness vector of a plane wave (assumption)
across a dense array at user-defined frequency bands.

By Chengxin Jiang, originally at May/20/2019 (Harvard)
    
Modified at Apr/2020 (ANU):
    1) follow the formula of Harmon et al., GJI (2010) for the beamforming with CCFs
    2) add the functionality of using sub-array data for the beaforming
'''

def geo2xyz(lat,lon):
    '''
    transform from geographic coordinates to spherical on surface

    PARAMETERS:
    -----------------
    lat: latitudes of the grids
    lon: longitude of the grids
    '''
    clon = np.mean(lon)
    clat = np.mean(lat)
    deg2km = 111
    xx = (lon-clon)*deg2km*np.cos(clat*np.pi/180)
    yy = (lat-clat)*deg2km

    return xx,yy

def get_station_geometry(sta_file,dt,npts,flag=False,source=None,rdist=None):
    '''
    this functions reads station info from the list, finds the relative coordinates
    and read one example waveform data for some basic parameters

    PAREMETERS:
    -----------------
    dpath: string, absolute path of the data 
    sta_file: string, full path for the station list
    comp: string, the component for CCFs to look at
    flag: boolen, True to do beaforming using subarrays
    source: string, central station to select subarrays, only needed when flag is True 
    rdist: np.float, all stations within this radisu of the source station

    RETURNS:
    -----------------
    sta: list of station names
    xx: np.array, relative coordinates in x direction (to central point)
    yy: np.array, same to xx but in y direction
    freqVec: np.array, 1D vector represent the frequency 
    npts: np.int, length of the waveform
    '''
    # loop through station
    locs = pd.read_csv(sta_file)
    sta  = list(locs['network']+'.'+locs['station'])
    lon  = np.array(locs['longitude']).astype(np.float32)
    lat  = np.array(locs['latitude']).astype(np.float32)

    # use subarray to do beamforming
    if flag:
        sta,lat,lon = make_new_stalist(sta,lon,lat,source,rdist)

    # transform the coordinate [cartersian coordinates is needed when array is large]
    xx,yy = geo2xyz(lat,lon)

    # frequency infor
    Nfft = int(next_fast_len(npts))
    freqVec = scipy.fftpack.fftfreq(Nfft,d=dt)[:(Nfft//2)]
    
    return sta,xx,yy,freqVec


def make_new_stalist(sta,lon,lat,source,rdist):
    '''
    extract stations located with rdist radius of the source station

    PARAMETERS:
    -----------------
    sta: list, original station list
    lon: np.array, 1D vector of station longitude
    lat: np.array, same to lon but for latitude
    source: string, targeted central station
    rdist: np.float, find all stations in certain radisu of source
    
    RETURNS:
    ------------
    sta_list: list, new station list
    nlon: np.array, 1D vector of new station longitude
    nlat: np.array, same to lon but for latitude
    '''
    # find index of central source station
    ista = sta.index(source)
    sta_list  = []
    nlon,nlat = [],[]
    for ii in range(len(sta)):
        dist,_,_ = obspy.geodetics.base.gps2dist_azimuth(lat[ista],lon[ista],lat[ii],lon[ii])
        if dist/1000 < rdist:
            sta_list.append(sta[ii])
            nlon.append(lon[ii])
            nlat.append(lat[ii])
    
    return sta_list,nlat,nlon

def load_spectrum_array(dpath,sta,dtype,comp,npts):
    '''
    this function finds all waveform data by the source and receveir list following the station.lst
    and loads the data and spectrum into arrays

    PARAMETERS:
    --------------
    dpath: string, abosolute path of the data
    sta:   list, station list 
    comp:  string, the component of CCFs to look at
    npts:  np.int, length of the waveform data

    RETURNS:
    ---------------
    freqVec: np.array, 1D vector of the frequency 
    data: np.array, 3D vectory representing all cross correlation
    spec: np.array, the corresponding spectrum of the data
    '''
    nsta = len(sta)

    # frequency information
    Nfft = int(next_fast_len(npts))

    # initalize the two arrays
    data = np.zeros(shape=(nsta,nsta,npts),dtype=np.float32)
    spec = np.zeros(shape=(nsta,nsta,Nfft//2),dtype=np.complex64)

    # loop through each source station
    for ii in range(nsta-1):
        for jj in range(ii+1,nsta):
            tfile1 = os.path.join(dpath,sta[ii]+'/'+str(sta[ii])+'_'+str(sta[jj])+'.h5')
            tfile2 = os.path.join(dpath,sta[jj]+'/'+str(sta[jj])+'_'+str(sta[ii])+'.h5')
            if os.path.isfile(tfile1):
                tfile = tfile1
            else:
                tfile = tfile2 
            
            # load the ASDF data
            with pyasdf.ASDFDataSet(tfile,mode='r') as ds:
                try:
                    tdata = ds.auxiliary_data[dtype][comp].data[:]
                    anpts = len(tdata)//2
                    data[ii,jj] = tdata[anpts:anpts+npts]*0.5+np.flip(tdata[anpts-npts:anpts],axis=0)*0.5
                    spec[ii,jj] = scipy.fftpack.fft(data[ii,jj],Nfft)[:Nfft//2]
                except Exception:
                    print("continue! cannot read %s "%tfile)
                    continue  
            
    return data,spec


@jit(nopython = True)
def sum_power(xx,yy,azim,slow,spec,freqVec):
    '''
    beamforming in frequency domain

    PARAMETERS:
    ----------------
    xx: np.array, x coordinates relative to the central point in the beam
    yy: np.array, y coordinates relative to the central point in the beam
    azim: np.float, azimuthal angle of the slowness vector
    slow: np.float, slowness value of the slowness vector
    spec: np.array, cross-correlation functions between the stations in source and receiver bins
    freqVec: np.int, length of window to stack the data for beamforming
    '''
    nsta = len(xx)
    npts = len(freqVec)

    # initialize one temporary array
    vector = np.zeros(npts,dtype=np.complex64)

    # array response functions
    xdis = xx*np.sin(azim*np.pi/180)+yy*np.cos(azim*np.pi/180)

    # loop through each frequency
    for ii in range(npts):
        ifreq = freqVec[ii]
        omega = ifreq*2*np.pi 
        resp  = np.exp(1j*omega*slow*xdis)
        
        # response function
        temp  = np.zeros(nsta,dtype=np.complex64)
        
        for jj in range(nsta):
            for kk in range(nsta):
                temp[jj] += resp[kk]*spec[kk,jj,ii]
            vector[ii] += temp[jj]*np.conj(resp[jj])

    beam = 10*np.log10(np.sum(np.real(vector)**2,axis=0))
    return beam


############################################
############## MAIN FUNCTIONS ##############
############################################

# rootpath for the data and output
rootpath = '/Volumes/Seagate/Kanto_basin/stacked'
sta_file  = 'station.lst'

# important parameters
freqmin = 0.1
freqmax = 1.0
subarray = True 
ssta_list = ['E.SKMM']
rdist = 15
comp  = ['ZZ','TT']
dtype = 'Allstack0linear'

# basic parameters about the CCFs
dt = 0.1
lag = 60
npts = int(lag/dt)+1

# beamformer parameters
slowmax = 2.5
dslow   = 0.05
nslow   = int(np.floor(slowmax/dslow))+1
slowness = np.linspace(0,slowmax,nslow)
nazim   = 91
azi_bin = np.linspace(0,360,nazim)

if not subarray:
    ssta_list = ['all']

for ssta in ssta_list:

    # get station geometry and some useful info
    sta,xx,yy,freq = get_station_geometry(sta_file,dt,npts,subarray,ssta,rdist)
    indx = np.where((freq>=freqmin) & (freq<=freqmax))[0]
    freqVec = freq[indx]
    num = len(sta)

    # load data and spec array
    data1,spec1 = load_spectrum_array(rootpath,sta,dtype,comp[0],npts)

    # perform the beamforming
    beampower1 = np.zeros(shape=(len(slowness),nazim),dtype=np.float32)

    for ii in range(nslow):
        slow = slowness[ii]
        for jj in range(nazim):
            azim  = azi_bin[jj]
            beam = sum_power(xx,yy,azim,slow,spec1[:,:,indx],freqVec)
            beampower1[ii,jj] = beam

    # plot the polar figure
    fig, (ax1,ax2) = plt.subplots(1,2,subplot_kw=dict(projection='polar'))
    x,y = np.meshgrid(np.radians(azi_bin),slowness)
    contour_inc = np.linspace(np.amin(beampower1),np.amax(beampower1),10)
    cm=ax1.contourf(x,y,beampower1,cmap='jet',levels=13,vmin=18,vmax=31)
    ax1.set_title('%s %s @%4.2f-%4.2f Hz'%(ssta,comp[0],freqmin,freqmax))
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1)
    cbar=fig.colorbar(cm, ax=ax1, shrink=0.4)
    cbar.ax.set_ylabel('power (dB)')

    # load data and spec array
    data2,spec2 = load_spectrum_array(rootpath,sta,dtype,comp[1],npts)

    # perform the beamforming
    beampower2 = np.zeros(shape=(len(slowness),nazim),dtype=np.float32)

    for ii in range(nslow):
        slow = slowness[ii]
        for jj in range(nazim):
            azim  = azi_bin[jj]
            beam = sum_power(xx,yy,azim,slow,spec2[:,:,indx],freqVec)
            beampower2[ii,jj] = beam

    x,y = np.meshgrid(np.radians(azi_bin),slowness)
    contour_inc = np.linspace(np.amin(beampower2),np.amax(beampower2),10)
    cm2=ax2.contourf(x,y,beampower2,cmap='jet',levels=13,vmin=18,vmax=31)
    ax2.set_title('%s %s @%4.2f-%4.2f Hz'%(ssta,comp[1],freqmin,freqmax))
    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    cbar1=fig.colorbar(cm2, ax=ax2, shrink=0.4)
    cbar1.ax.set_ylabel('power (dB)')
    fig.tight_layout()

    # save the plot
    if not os.path.isdir(os.path.join(rootpath,'figures')):
        os.mkdir(os.path.join(rootpath,'figures'))
        
    if not subarray:
        fout = '{0:s}/All_{1:4.2f}_{2:4.2f}_{3:s}.pdf'.format(rootpath+'/figures',freqmin,freqmax,comp)
    else:
        fout = '{0:s}/figure3_{1:s}_{2:4.2f}_{3:4.2f}_{4:d}km_{5:s}.pdf'.format(rootpath,ssta,freqmin,freqmax,rdist,comp[0])
    fig.savefig(fout,format='pdf', dpi=300)
    plt.close('all')
