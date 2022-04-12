#!/bin/tcsh -f

#---this script tries to synthesize the seismogram for both R and L waves----
#---used to infer the amplitude of RR,ZZ components------
# make sure CPS is installed on your local computer before running this script

set code_path = '/Users/chengxin/progs/cps/bin'
\rm -f B*.sac

set mod = $1
set station = `echo $mod | awk -F_ '{print $2}' | awk -F. '{print $1}'`
set dfile = dfile.dat
set pfile = disp.dat

#----check input files----
if ($#argv<1) then
  echo "usage:code+mod.dat"
  exit
endif
cp $mod model.dat

#---some common parameters for synthetic----
set dt = 0.1
set npts = 500
set nmod = 2
set t0   = 0
set verd = 0

set nnn = 1
# one random distance array
foreach dist (2.7 3.1 4.5 5.2 5.4 6.8 7.0 7.2 7.4 7.7 8.1 9.2 9.5 9.8 10.0 )

echo $dist $dt $npts $t0 $verd > $dfile
echo 1 | awk 'END{for (ii=0.5;ii<8;ii=ii+0.5) print ii}'> $pfile

#---start the running----
${code_path}/sprep96 -M model.dat -DT $dt -HR 0 -HS 0 -NPTS $npts -L -R -NMOD $nmod -d $dfile
${code_path}/sdisp96 -v > sdisp96.out
${code_path}/sregn96 > sregn96.out
${code_path}/slegn96 > slegn96.out
${code_path}/sdpegn96 -L -U -ASC
${code_path}/sdpegn96 -R -U -ASC
${code_path}/spulse96 -d $dfile -V -t -l 3 -D -OD -EX  | ${code_path}/fprof96
${code_path}/spulse96 -d $dfile -V -t -l 3 -D -OD -EX  > file96 
${code_path}/f96tosac -B file96

set zfile = `echo $nnn $station | awk '{printf("%s_Z%d.sac",$2,$1)}'`
set rfile = `echo $nnn $station | awk '{printf("%s_R%d.sac",$2,$1)}'`

mv B00109ZEX.sac results/$zfile
mv B00110REX.sac results/$rfile
set nnn = `echo "$nnn+1"|bc`
end

# generate a 1D model file
awk 'NR>12{print $1","$2","$3","$4}' model.dat > results/1D_model.lst
echo "th,vp,vs,rho" > temp.dat
cat results/1D_model.lst >> temp.dat
mv temp.dat results/1D_model.lst
