#!/bin/bash
#
# 2D SCF-Net: fault polylines of the Opunake image from a trained network (src/fault_surfaces.py), with the parameters
# grouped by the property of the faults that they control. Run it in the 2d directory after run_train.sh. Results in
# $out, for the input mode $mode:
#
#   opunake.<mode>.fsem/.fdip       fault probability and dip (the dip wherever the probability is above 0, as the
#                                   fault cells need it at low probabilities)
#   opunake.<mode>.faults.png       fault polylines on the image, colored by $color
#   opunake.<mode>.faults.vtk       fault polylines with the attributes fault, probability and dip (ParaView)
#   opunake.<mode>.faults.h5        the same in HDF5: nvertex, x1, x2 (samples), fault, probability, dip, and nsegment
#                                   and segments (0-based indices of the vertices)
#   opunake.<mode>.events.vtk       events of the seismicity as balls, with the fault probability at the event (ParaView)
#   opunake.<mode>.volume.vti       image, seismicity location image, and fault probability and dip, in the frame of
#                                   the faults (ParaView)
#   opunake.<mode>.faults.txt, .json
#                                   cells of the faults, and a summary of every fault
#
set -e
[ -f generate_data.py ] || { echo "Run this script in the 2d directory"; exit 1; }

model=result/last.ckpt          # trained network (result_opunake/last.ckpt for the fine-tuned network)
mode=image                      # input mode: image, both, or meq
out=result/surfaces             # results

# Coverage: how weak a fault can be, and how short
lower=0.2                       # least probability of a fault cell; lower (e.g., 0.1) extends faults into their weak parts
upper=0.5                       # least probability of the most confident cell of a fault; lower (e.g., 0.3) adds faults
                                # that are weak everywhere, and more false faults
min_cells=20                    # shortest fault, in cells; lower keeps shorter faults

# Continuity: when cells and pieces belong to the same fault
max_gap=5                       # longest gap, in samples, across which pieces of a fault merge; higher joins more pieces,
                                # and may join different faults; 0: no merging
max_planar=1.0                  # largest distance, in samples, of a linked cell from the line of the other; higher links
                                # curved and noisy faults
max_ddip=10                     # largest difference of dip between linked cells, degrees; higher links faults that bend

# Shape of the polylines
smooth_surface=4                # radius, in samples, of the local line fits that smooth the faults; higher is smoother,
                                # and straightens the faults; 0: no smoothing

# Noisy fault probability (e.g., of the seismicity-only mode): optimal surface voting favors continuous faults over
# isolated responses; it needs lower=0.65 and upper=0.8
vote=""                         # "--vote" to vote

# Figure: attribute that colors the faults
color=dip                       # fault, probability, or dip

# Seismicity: the original event locations, if available (used by the inference and drawn as balls); otherwise, the
# seismicity location image test/opunake.bin.meq, whose maxima give the events (events closer than about two samples
# merge)
events=""                       # text or CSV file of event locations: x1 and x2 grid coordinates (from 0) in its first
                                # columns, and attributes (e.g., the magnitude) in further columns, the first of which
                                # (the third column) colors the balls; a first line of names is allowed
event_radius=1.0                # radius of the balls, samples

mkdir -p $out

if [ -n "$events" ]; then
	seismicity="--input_events=$events"
	surfaces="--events=$events"
else
	seismicity="--input_meq=test/opunake.bin.meq"
	surfaces="--meq=test/opunake.bin.meq"
fi

# Inference, block by block, with the dip wherever the probability is above 0
python ../src/main2.py --model=$model --input=test/opunake.bin $seismicity \
	--n1=256 --n2=1024 --mode=$mode --b1=256 --b2=256 --p1=64 --p2=64 \
	--fault_threshold=0 --output=$out/opunake.$mode

# Fault polylines
python ../src/fault_surfaces.py --fsem=$out/opunake.$mode.fsem --fdip=$out/opunake.$mode.fdip \
	--n1=256 --n2=1024 --image=test/opunake.bin --output=$out/opunake.$mode \
	--lower=$lower --upper=$upper --min_cells=$min_cells \
	--max_gap=$max_gap --max_planar=$max_planar --max_ddip=$max_ddip \
	--smooth_surface=$smooth_surface $vote --color=$color --h5 --volume \
	$surfaces --event_radius=$event_radius
