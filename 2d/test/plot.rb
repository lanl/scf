
# Plotting of the results needs pymplot at github.com/lanl/pymplot

opts = "-size1=3 -size2=9 -n1=256 -tick1d=50 -mtick1=4 -tick2d=100 -mtick2=4 -label1='Z (grid number)' -label2='X (grid number)' -interp=gaussian "

# Plot synthetic data validation
for i in 1..10

    system "x_showmatrix -background=../dataset_finetune/data_train/#{i}_img.bin -in=../dataset_finetune/data_train/#{i}_meq.bin #{opts} -backcolor=binary -backclip=2 -alphas=0.1:0,0.11:1 -interp=none -color=jet -legend=y -unit='Source Location Probability' -out=#{i}.pdf &"

    system "x_showmatrix -background=../dataset_finetune/data_train/#{i}_meq.bin -backinterp=gaussian -in=../dataset_finetune/target_train/#{i}_fdip.bin #{opts} -interp=none -cmin=0.1 -cmax=0.7 -color=jet -ctruncend=0.9 -alphas=0.1:0,0.101:1 -legend=y -unit='Fault Dip ($\\times 180$ deg.)' -backcolor=binary -backcmin=0 -backcmax=1 -out=#{i}_fault.pdf &"

    puts i

end

label = "opunake"

# The trained models are not included. You need to train them first.
mlopts = " --input=opunake.bin --n1=256 --n2=1024 "

if true

    system "python ../src/main2.py --model=../result_mtl/last.ckpt --net=mtl --output=predict_mtl #{mlopts}"

    system "python ../src/main2.py --model=../result_scf/last.ckpt --net=scf --output=predict_scf #{mlopts}"

    system "python ../src/main2.py --model=../result_scf_finetune/last.ckpt --net=scf --output=predict_scf_finetune #{mlopts} "

end

# Plot image + seismicity
system "x_showmatrix -background=opunake.bin -in=opunake.bin.meq #{opts} -backcolor=binary -backclip=0.5 -alphas=0.1:0,0.11:1 -color=jet -ctruncend=0.9 -legend=y -unit='Source Location Probability' -out=#{label}_img_meq.pdf &"

# Plot image + fualt dip
opts = opts + " -interp=none -cmin=0.2 -cmax=0.7 -color=jet -ctruncend=0.9 -alphas=0.2:0,0.201:1 -legend=y -unit='Fault Dip ($\\times 180$ deg.)' "
for i in ['mtl', 'scf', 'scf_finetune']

    system "x_showmatrix -background=opunake.bin -in=predict_#{i}.fdip #{opts} -backcolor=binary -backclip=0.5  -out=#{label}_img_fdip_#{i}.pdf &"

    system "x_showmatrix -background=opunake.bin.meq -backinterp=gaussian -in=predict_#{i}.fdip #{opts} -backcolor=binary -backcmin=0 -backcmax=1 -out=#{label}_meq_fdip_#{i}.pdf &"

end

# Plot seismicity + fault dip
for i in ['mtl', 'scf', 'scf_finetune']

	opts = opts + " -x1beg=150 -x2beg=540 -x2end=750 -size1=3 -size2=6 "
    system "x_showmatrix -background=opunake.bin -in=predict_#{i}.fdip #{opts} -backcolor=binary -backclip=0.5 -out=#{label}_img_fdip_#{i}_zoom.pdf &"

    system "x_showmatrix -background=opunake.bin.meq -backinterp=gaussian -in=predict_#{i}.fdip #{opts} -backcolor=binary -backcmin=0 -backcmax=1 -out=#{label}_meq_fdip_#{i}_zoom.pdf &"

end
