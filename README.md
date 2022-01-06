# Scripts to reproduce several important figures for Jiang & Denolle, 2022 (submitted to JGR). 
This respo contains some python/shell scripts to reproduce several important figures for Jiang & Denolle, 2022. The scripts require the cross-correlation functions and/or the velocity model derived by Jiang & Denolle, 2022. Both files are achieved at [Zenos](URL) and are publically accessible. Detailed description, usage and input files for each script are provided below. 

Contact Chengxin Jiang (chengxin.jiang1@anu.edu.au) if you have any questions or any bugs to report.  

## script of figure3\_beamforming\_CCFs.py
* Description: a python script to make beamforming calculations upon derived cross-correlation functions
* Inputs: the cross correlation functions between all MeSO-net stations and the station list
* Usage: python figure3\_beamforming\_CCFs.py
<img src="figures/figure3.jpg" width="800" height="400">

## script of figure4\_moveout\_matrix.py
* Description: a python script to reproduce the moveout plot for figure 4
* Inputs: the cross correlation functions between all MeSO-net stations
* Usage: python figure4\_moveout\_matrix.py
<img src="figures/figure4_moveout.jpg" width="800" height="500">

## script of figure6\_dispersion.py
* Description: a python script to reproduce the phase diagram using raw cross correlation functions 
* Inputs: the cross correlation functions between all MeSO-net stations and the station list
* Usage: python figure6\_dispersion.py
<img src="figures/figure6_dispersion.jpg" width="800" height="400">

## script of figure10\_plot\_iso.csh
* Description: a C shell script to plot the mapviews of the Vs model at 0.5, 1, 2 and 2.5 km depth, respectively
* Inputs: station list; 3D velocity model (model\_Kanto\_0.01inc.dat); output file name
* Usage: csh figure10\_plot\_iso.csh model\_Kanto\_0.01inc.dat
<img src="figures/figure10_vs.jpg" width="800" height="400">

## script of figure12\_plot\_aniso.csh
* Description: a C shell script to plot the mapviews of the anisotropy model at 0.5, 1, 2 and 2.5 km depth, respectively
* Inputs: station list; 3D velocity model (model\_Kanto\_0.01inc.dat); output file name
* Usage: csh figure12\_plot\_aniso.csh model\_Kanto\_0.01inc.dat
<img src="figures/figure12_aniso.jpg" width="800" height="400">
