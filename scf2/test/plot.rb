
# Plotting of the results needs pymplot at github.com/lanl/pymplot




opts = "-size1=3 -size2=9 -n1=256 -tick1d=50 -mtick1=4 -tick2d=100 -mtick2=4 -label1='Z (grid number)' -label2='X (grid number)' -interp=gaussian "

label = "opunake"

# The trained models are not included. You need to train them first. 
if true

    system "python ../srr/nomeq/main2_infer.py --model=../result_nomeq/epoch=90.ckpt --input=opunake.bin --n1=256 --n2=1024 --output=predict_nomeq "

    system "python ../src/meq/main2_infer.py --model=../result_meq/epoch=99.ckpt --input=opunake.bin --n1=256 --n2=1024 --output=predict_meq "

end

system "x_showmatrix -background=opunake.bin -in=opunake.bin.meq #{opts} -backcolor=binary -backclip=0.5 -alphas=0.1:0,0.11:1 -color=jet -legend=y -unit='Source Location Probability' -out=./#{label}_img_meq.png,./#{label}_img_meq.pdf &"




opts = opts + " -interp=none -cmin=0.2 -cmax=0.7 -color=jet -ctruncend=0.9 -alphas=0.2:0,0.201:1 -legend=y -unit='Fault Dip ($\\times 180$ deg.)' "
for i in ['nomeq', 'meq']

    system "x_showmatrix -background=opunake.bin -in=predict_#{i}.fdip #{opts} -backcolor=binary -backclip=0.5  -out=./#{label}_img_fdip_#{i}.png,./#{label}_img_fdip_#{i}.pdf &"

    system "x_showmatrix -background=opunake.bin.meq -backinterp=gaussian -in=predict_#{i}.fdip #{opts} -backcolor=binary -backcmin=0 -backcmax=1 -out=./#{label}_meq_fdip_#{i}.png,./#{label}_meq_fdip_#{i}.pdf &"

end

for i in ['nomeq', 'meq']

	opts = opts + " -x1beg=150 -x2beg=700 -size1=3 -size2=6 "
    system "x_showmatrix -background=opunake.bin -in=predict_#{i}.fdip #{opts} -backcolor=binary -backclip=0.5 -out=./#{label}_img_fdip_#{i}_zoom.png,./#{label}_img_fdip_#{i}_zoom.pdf &"

    system "x_showmatrix -background=opunake.bin.meq -backinterp=gaussian -in=predict_#{i}.fdip #{opts} -backcolor=binary -backcmin=0 -backcmax=1 -out=./#{label}_meq_fdip_#{i}_zoom.png,./#{label}_meq_fdip_#{i}_zoom.pdf &"

end