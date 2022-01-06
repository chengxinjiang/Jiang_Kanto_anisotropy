#!/bin/csh -f

########################
# extract Vs models at specified depth
# from a 3D model and plot the models
# by Chengxin Jiang
########################

gmtset BASEMAP_TYPE plain
gmtset MEASURE_UNIT inch
gmtset HEADER_FONT = Times-Roman
gmtset HEADER_FONT_SIZE = 10p
gmtset ANNOT_FONT_SIZE_PRIMARY = 10p
gmtset ANNOT_FONT_SIZE_SECONDARY    = 10p
gmtset HEADER_OFFSET = -0.15
gmtset PLOT_DEGREE_FORMAT      = ddd.x

#----input file of the 3D model-----
set infn = $1
set outfn = aniso.ps

if ($#argv<1) then
  echo "usage:code+3D_model"
  exit
endif

#---extract 2D model at each depth---
set infn1  = depth_0.5_ani.dat
set infn2  = depth_1_ani.dat
set infn3  = depth_2_ani.dat
set infn4  = depth_2.5_ani.dat

awk 'NR>1{if ($3==0.5 && $6!="NaN") print $1,$2,$6,$7}' $infn > $infn1
awk 'NR>1{if ($3==1.0 && $6!="NaN") print $1,$2,$6,$7}' $infn > $infn2
awk 'NR>1{if ($3==2.0 && $6!="NaN") print $1,$2,$6,$7}' $infn > $infn3
awk 'NR>1{if ($3==2.5 && $6!="NaN") print $1,$2,$6,$7}' $infn > $infn4

#----for plotting---
set range = 139.3/140.7/35.3/36.2
set tic = a0.5f0.1/a0.5f0.1WSen
set tic1 = a0.5f0.1/a0.5f0.1wSen
set size = 2.4
set x = 3.4
set y = 2.6
set cpt = temp.cpt
makecpt -Cpolar -T-8/8/0.5 -Z -D -I > $cpt

######### 0.5 km depth #########

########
# one can add the topography component here, which is removed due to the large size of topo.grd
########
#grdsample $tfile -Gtopo.grd -R$range -I0.1m -V
#grdgradient topo.grd -Gtopo.gradient -A240 -Nt -V
#grdimage topo.grd -R$range -JM$size -B$tic -Ctopo.cpt -Itopo.gradient -K -V -P -Y6 -X1 > $outfn

pscoast -JM$size -R$range -B${tic}:."(a) 0.5 km": -Dh -W1 -V -N1 -K -L140.5/35.4/35.4/6 -K -Sgray -Ggray -P -Y6 -X1 > $outfn
awk '{if ($3!="NaN") print $1,$2,$3}' $infn1 | psxy -JM -R -Sc0.025 -C$cpt -V -K -O -P >> $outfn
awk -F, 'NR>1{print $4,$3}' station.lst | psxy -J -R -Sc0.03 -W1.5/100/100/100  -O -P -K >> $outfn
psscale -C$cpt -D2.55/0.95/1.9/0.12 -Ba5f1 -O -K >> $outfn

########### 1.0 km depth ############
pscoast -JM$size -R$range -B${tic1}:."(b) 1.0 km":  -Dh -W1 -V -N1 -K -L140.5/35.4/35.4/6 -Sgray -O -Ggray -X$x >> $outfn
awk '{if ($3!="NaN") print $1,$2,$3}' $infn2 | psxy -JM -R -Sc0.025 -C$cpt -V -K -O >> $outfn
awk -F, 'NR>1{print $4,$3}' station.lst | psxy -J -R -Sc0.03 -W1.5/100/100/100  -O -P -K >> $outfn
psscale -C$cpt -D2.55/0.95/1.9/0.12 -Ba5f1 -O -K >> $outfn

######### 2.0 km depth #########
pscoast -JM$size -R$range -B${tic}:."(c) 2.0 km": -Dh -W1 -V -N1 -K -L140.5/35.4/35.4/6 -O -Sgray -Ggray -X-$x -Y-$y >> $outfn
awk '{if ($3!="NaN") print $1,$2,$3}' $infn3 | psxy -JM -R -Sc0.025 -C$cpt -V -K -O >> $outfn
awk -F, 'NR>1{print $4,$3}' station.lst | psxy -J -R -Sc0.03 -W1.5/100/100/100  -O -P -K >> $outfn
psscale -C$cpt -D2.55/0.95/1.9/0.12 -Ba5f1 -O -K >> $outfn

########### second depth ############
pscoast -JM$size -R$range -B${tic1}:."(d) 2.5 km": -Dh -W1 -V -N1 -K -L140.5/35.4/35.4/6 -O -Sgray -Ggray -X$x >> $outfn
awk '{if ($3!="NaN") print $1,$2,$3}' $infn4 | psxy -JM -R -Sc0.025 -C$cpt -V -K -O >> $outfn
awk -F, 'NR>1{print $4,$3}' station.lst | psxy -J -R -Sc0.03 -W1.5/100/100/100  -O -P -K >> $outfn
echo 1 | awk '{print "139.6 35.85";print "140.5 35.7"}' | psxy -J -R -W1.5p,255/0/255,5_2:0p -K -O -P -V >> $outfn
echo 1 | awk '{print "139.5 35.6";print "140.3 35.9"}' | psxy -J -R -W1.5p,255/0/255,5_2:0p -K -O -P -V >> $outfn
psscale -C$cpt -D2.55/0.95/1.9/0.12 -Ba5f1 -O >> $outfn

\rm temp* depth*.dat
