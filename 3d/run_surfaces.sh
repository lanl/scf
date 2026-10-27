#!/bin/bash
#
# 3D SCF-Net: fault surfaces of the F3 volume from a trained network (src/fault_surfaces.py), with the parameters
# grouped by the property of the faults that they control. Run it in the 3d directory after run_train.sh. Results in
# $out, for the input mode $mode:
#
#   f3.<mode>.fsem/.fdip/.fstrike   fault probability, dip and strike (the dip and strike wherever the probability is
#                                   above 0, as the fault cells need them at low probabilities)
#   f3.<mode>.faults.vtk            fault surfaces with the attributes fault, probability, dip and strike (ParaView)
#   f3.<mode>.faults.h5             the same in HDF5: nvertex, x1, x2, x3 (samples), fault, probability, dip, strike,
#                                   and ntriangle and triangles (0-based indices of the vertices)
#   f3.<mode>.events.vtk            events of the seismicity as balls, with the fault probability at the event (ParaView)
#   f3.<mode>.volume.vti            image, seismicity location image, and fault probability, dip and strike, in the
#                                   frame of the faults (ParaView)
#   f3.<mode>.faults.png            fault surfaces with image slices, colored by $color
#   f3.<mode>.faults.txt, .json     cells of the faults, and a summary of every fault
#
set -e
[ -f generate_data.py ] || { echo "Run this script in the 3d directory"; exit 1; }

model=result/last.ckpt          # trained network
mode=image                      # input mode: image, both, or meq
out=result/surfaces             # results

# Coverage: how weak a fault can be, and how small
lower=0.2                       # least probability of a fault cell; lower (e.g., 0.1) extends faults into their weak parts
upper=0.5                       # least probability of the most confident cell of a fault; lower (e.g., 0.3) adds faults
                                # that are weak everywhere, and more false faults
min_cells=200                   # smallest fault, in cells; lower (e.g., 50) keeps smaller faults

# Continuity: when cells and pieces belong to the same fault
max_gap=5                       # longest gap, in samples, across which pieces of a fault merge (and are joined in the
                                # surface); higher joins more pieces, and may join different faults; 0: no merging
max_planar=1.0                  # largest distance, in samples, of a linked cell from the plane of the other; higher links
                                # curved and noisy faults
max_ddip=10                     # largest difference of dip between linked cells, degrees; higher links faults that bend
max_dstrike=30                  # largest difference of strike between linked cells, degrees

# Shape of the surfaces
smooth_surface=4                # radius, in samples, of the local plane fits that smooth the faults; higher is smoother,
                                # and flattens the curvature of the faults; 0: no smoothing
max_hole=50                     # largest hole inside a fault surface that is filled, in square samples

# Noisy fault probability (e.g., of the seismicity-only mode): optimal surface voting favors continuous faults over
# isolated responses; it scores the faults relative to the strongest one, so it drops small faults, and needs
# lower=0.65 and upper=0.8
vote=""                         # "--vote" to vote

# Figure: attribute that colors the faults
color=dip                       # fault, probability, dip, or strike

# Seismicity: the original event locations, if available (used by the inference and drawn as balls); otherwise, the
# seismicity location image test/dutch.meq, whose maxima give the events (events closer than about two samples merge)
events=""                       # text or CSV file of event locations: x1, x2 and x3 grid coordinates (from 0) in its
                                # first columns, and attributes (e.g., the magnitude) in further columns, the first of
                                # which (the fourth column) colors the balls; a first line of names is allowed
event_radius=1.0                # radius of the balls, samples

mkdir -p $out

# F3 volume of 128 x 256 x 256 samples, as in run_valid.sh
if [ ! -f test/dutch.img ]; then
	cat test/f3_img.z01 test/f3_img.z02 test/f3_img.z03 test/f3_img.zip > test/f3_img_joined.zip
	unzip -o -q test/f3_img_joined.zip -d test || true
	rm -f test/f3_img_joined.zip
fi
[ -f test/dutch.meq ] || unzip -o -q test/f3_meq.zip -d test
[ "$(stat -c %s test/dutch.img)" = 33554432 ] || { echo "test/dutch.img is not 128 x 256 x 256 floats"; exit 1; }

if [ -n "$events" ]; then
	seismicity="--input_events=$events"
	surfaces="--events=$events"
else
	seismicity="--input_meq=test/dutch.meq"
	surfaces="--meq=test/dutch.meq"
fi

# Inference, block by block, with the dip and strike wherever the probability is above 0
python ../src/main3.py --model=$model --input=test/dutch.img $seismicity \
	--n1=128 --n2=256 --n3=256 --mode=$mode --b1=128 --b2=128 --b3=128 --p1=32 --p2=32 --p3=32 \
	--fault_threshold=0 --output=$out/f3.$mode

# Fault surfaces
python ../src/fault_surfaces.py --fsem=$out/f3.$mode.fsem --fdip=$out/f3.$mode.fdip --fstrike=$out/f3.$mode.fstrike \
	--n1=128 --n2=256 --n3=256 --image=test/dutch.img --output=$out/f3.$mode \
	--lower=$lower --upper=$upper --min_cells=$min_cells \
	--max_gap=$max_gap --max_planar=$max_planar --max_ddip=$max_ddip --max_dstrike=$max_dstrike \
	--smooth_surface=$smooth_surface --max_hole=$max_hole $vote --color=$color --volume --h5 \
	$surfaces --event_radius=$event_radius
