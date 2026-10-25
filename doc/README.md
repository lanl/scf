# Use of SCF-Net

`src/main2.py` (2D) and `src/main3.py` (3D) train a network (default), validate a trained network on the validation set (`--check=<checkpoint>`), or apply it to an image or a volume (`--model=<checkpoint>`). One network detects faults from the image and the seismicity (`both`), from the image only (`image`), or from the seismicity only (`meq`). `python src/main2.py --help` lists all options.

## Data

All arrays are raw float32 files with the first (depth) axis fastest. The training and validation sets, as written by `2d/generate_data.py` and `3d/generate_data.py`, contain `<i>_img.bin` (seismic image) and `<i>_meq.bin` (seismicity location image) in `data_train` and `data_valid`, and `<i>_fsem.bin` (fault mask), `<i>_fdip.bin` (dip / 180), `<i>_fstrike.bin` (strike / 180, 3D only), and `<i>_fsem_meq.bin` (the faults that the seismicity indicates, the label of the seismicity-only mode) in `target_train` and `target_valid`. Without `<i>_fsem_meq.bin`, the label of the seismicity-only mode is the part of the faults within four samples of the seismicity.

The data generation runs on several MPI processes with `mpirun -np <n> python generate_data.py --mpi ...` (or `srun -n <n> ...`), with one or two OpenMP threads per process (`OMP_NUM_THREADS`). Every model depends only on its index and `--seed`, so the data do not depend on the number of processes.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--n1`, `--n2`, `--n3` | 256 (2D), 128 (3D) | numbers of samples of the data, or of the input of inference |
| `--ntrain`, `--nvalid` | | numbers of training and validation samples |
| `--p_image`, `--p_meq` | `0.3`, `0.2` | shares of the training samples with the image only and with the seismicity only |
| `--crop1`, `--crop2`, `--crop3` | 128 (2D), 64 (3D) | smallest random training crop along every axis |
| `--crop_max` | sample size | largest random training crop along every axis, to fit the GPU memory |
| `--w_dip`, `--w_consist` | `1`, `1` | weights of the dip (and strike) loss and of the patch-consistency loss |
| `--epochs`, `--warmup`, `--lr`, `--bn_freeze` | `60`, `5`, `2e-4`, `6` | epochs, warmup epochs, learning rate, and final epochs with fixed batch-normalization statistics |
| `--gpus_per_node`, `--nodes`, `--precision` | `1`, `1`, `32` | GPUs per node, nodes, and precision of training (`bf16-mixed` needs GPUs with bfloat16, e.g., A100 or H100) |
| `--finetune`, `--resume` | | checkpoint that initializes the network, or from which the training resumes |
| `--mode` | `all` | inference from the `image` only, the seismicity only (`meq`), `both`, or `all` three |
| `--input`, `--input_meq`, `--input_events` | | image, seismicity location image, or event locations (a text file of grid coordinates); either input can be missing |
| `--b1`, `--b2`, `--b3`, `--p1`, `--p2`, `--p3` | whole image, 64 (2D), 32 (3D) | block sizes and overlaps of block-wise inference |
| `--scale` | `rms` | amplitude normalization of the image: `rms` of the whole image, or `agc` (local) |

With several GPUs, the training uses distributed data parallelism: the batch size is per GPU, and batch normalization uses the batches of all GPUs. Without SLURM, Lightning starts one process per GPU. With SLURM, start one task per GPU, also on one node, e.g., `srun --nodes=2 --ntasks-per-node=4 python src/main3.py --gpus_per_node=4 --nodes=2 ...`.

## Inference

For example, the faults of an image of 256 x 1024 samples from the image only, both inputs, and the seismicity only, block by block,

```
python src/main2.py --model=2d/result/last.ckpt --input=image.bin --input_meq=image.bin.meq \
    --n1=256 --n2=1024 --mode=all --b1=256 --b2=256 --p1=64 --p2=64 --output=predict
```

writes `predict.image.fsem`, `predict.both.fsem`, and `predict.meq.fsem` (fault probability) and the corresponding `.fdip` files (fault dip divided by 180 degrees, times the fault probability); in 3D, `src/main3.py` also writes `.fstrike` files. The image is normalized by its RMS once, the blocks start on the pooling grid of the network and are blended with a raised-cosine taper, and the image stays in the CPU memory while every block goes to the GPU, so a large volume gives the same faults as whole-volume inference.
