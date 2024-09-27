
label = "dutch"

#system "cp -rp dutch_select.bin dutch.img"

if true

    system "python ../src/nomeq/main3_infer.py --model=../result_nomeq/epoch=49.ckpt --input=dutch.img --n1=128 --n2=256 --n3=256 --output=#{label}_nomeq "

    system "python ../src/meq/main3_infer.py --model=../result_meq/epoch=51.ckpt --input=dutch --n1=128 --n2=256 --n3=256 --output=#{label}_meq "

end

opts = "-size1=3 -size2=6 -size3=6 -n1=128 -n2=256 -tick1d=50 -mtick1=4 -tick2d=50 -mtick2=4 -tick3d=50 -mtick3=4 -label1='Z' -label2='Y' -label3='X' -slice1=110 -slice2=125 "

system "x_showslice -background=dutch.img -in=dutch.meq #{opts} -backcolor=binary -backclip=2 -alphas=0.1:0,0.11:1 -color=jet -ctruncend=0.9 -tr=img_meq.png -out=./#{label}_img_meq.pdf -legend=y -unit='Source Location Probability' &"

abort

for i in ['nomeq', 'meq']

    system "x_showslice -background=dutch.img -in=dutch_#{i}.fdip #{opts} -backcolor=binary -backclip=2 -alphas=0.2:0,0.21:1 -color=jet -legend=y -unit='Fault Dip ($\\times 180$ deg.)' -cmin=0.2 -cmax=0.7 -out=./#{label}_img_fdip_#{i}.pdf &"

    system "x_showslice -background=dutch.img -in=dutch_#{i}.fstrike #{opts} -backcolor=binary -backclip=2 -alphas=0.0:0,0.02:1 -color=jet -legend=y -unit='Fault Strike ($\\times 180$ deg.)' -cmin=0 -cmax=1 -ctruncbeg=0.1 -ctruncend=0.9 -out=./#{label}_img_fstrike_#{i}.pdf &"
    
    system "x_showslice -background=dutch.meq -in=dutch_#{i}.fdip #{opts} -backcolor=binary -backcmin=0 -backcmax=1 -alphas=0.2:0,0.21:1 -color=jet -legend=y -tr=fdip_#{i}.png -unit='Fault Dip ($\\times 180$ deg.)' -cmin=0.2 -cmax=0.7 -out=./#{label}_meq_fdip_#{i}.pdf &"

end
