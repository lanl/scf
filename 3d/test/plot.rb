
label = "dutch"

# The seismic image is too big, so I split it into three smaller zip files. Please unzip img.zip and meq.zip to obtain the file.

# The seismic image is selected and resampled from the open-source North Sea F3 data available at [F3](https://wiki.seg.org/wiki/F3_Netherlands).

# The seismicity location image is created based on preliminary fault detection result.


n1 = 128
n2 = 256
n3 = 256

slice1 = 111
slice2 = 116
slice3 = 151

b1 = 128
b2 = 128
b3 = 128
p1 = 45
p2 = 45
p3 = 45

# Do inference
if true

    system "python ../src/main3.py --model=../result_mtl/last.ckpt --input=dutch --net=mtl --n1=#{n1} --n2=#{n2} --n3=#{n3} --output=#{label}_mtl --gpus_per_node=1 --b1=#{b1} --b2=#{b2} --b3=#{b3} --p1=#{p1} --p2=#{p2} --p3=#{p3} "

    system "python ../src/main3.py --model=../result_scf/last.ckpt --input=dutch --net=scf --n1=#{n1} --n2=#{n2} --n3=#{n3} --output=#{label}_scf --gpus_per_node=1 --b1=#{b1} --b2=#{b2} --b3=#{b3} --p1=#{p1} --p2=#{p2} --p3=#{p3} "

end

# Plot image + seismicity
opts = "-size1=2.5 -size2=5 -size3=5 -n1=#{n1} -n2=#{n2} -tick1d=50 -mtick1=4 -tick2d=50 -mtick2=4 -tick3d=50 -mtick3=4 -label1='Z (Grid Number)' -label2='Y (Grid Number)' -label3='X (Grid Number)' -slice1=#{slice1} -slice2=#{slice2} -slice3=#{slice3} "

system "x_showslice -background=dutch.img -in=dutch.meq #{opts} -backcolor=binary -backclip=2 -alphas=0.1:0,0.11:1 -color=jet -ctruncend=0.9 -out=./#{label}_img_meq.pdf -legend=y -unit='Source Location Probability' &" # -tr=img_meq.png

# Plot image/seismicity + fault dip/strike
for i in ['mtl', 'scf']

    system "x_showslice -background=dutch.img -in=dutch_#{i}.fdip #{opts} -backcolor=binary -backclip=2 -alphas=0.3:0,0.31:1 -color=jet -legend=y -unit='Fault Dip ($\\times 180$ deg.)' -cmin=0.3 -cmax=0.7 -ld=0.1 -lmtick=9 -out=#{label}_img_fdip_#{i}.pdf &" # -tr=img_fdip_#{i}.png

    system "x_showslice -background=dutch.meq -in=dutch_#{i}.fdip #{opts} -backcolor=binary -backcmin=0 -backcmax=1 -alphas=0.3:0,0.31:1 -color=jet -legend=y -unit='Fault Dip ($\\times 180$ deg.)' -cmin=0.3 -cmax=0.7 -ld=0.1 -lmtick=9 -out=#{label}_meq_fdip_#{i}.pdf &" # -tr=img_fdip_#{i}.png

    system "x_showslice -background=dutch.img -in=dutch_#{i}.fstrike #{opts} -backcolor=binary -backclip=2 -alphas=0.0:0,0.02:1 -color=jet -legend=y -unit='Fault Strike ($\\times 180$ deg.)' -cmin=0 -cmax=1 -ctruncbeg=0.25 -ctruncend=0.9 -ld=0.2 -lmtick=9 -out=#{label}_img_fstrike_#{i}.pdf &" # -tr=img_fstrike_#{i}.png

    system "x_showslice -background=dutch.meq -in=dutch_#{i}.fstrike #{opts} -backcolor=binary -backcmin=0 -backcmax=1 -alphas=0.0:0,0.02:1 -color=jet -legend=y -unit='Fault Strike ($\\times 180$ deg.)' -ctruncbeg=0.25 -ctruncend=0.9 -cmin=0 -cmax=1 -ld=0.2 -lmtick=9 -out=#{label}_meq_fstrike_#{i}.pdf &" # -tr=img_fstrike_#{i}.png

end
