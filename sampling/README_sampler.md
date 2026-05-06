# README for sampling


main file: `sampler_analytical_RC3D.py`
run this file to start the sampler. 


## Flags available at the start of the script: 

`CHUNKED` 
This should be used for runs on Euler.
Set CHUNKED=True (recommended) to iterate over the dataset in CHUNK_SIZE
pieces — keeps RAM usage ~constant regardless of total sample count.
Set False only for small runs where all data fits comfortably in memory.
Ignored when SAMPLE_2D=True (2D path uses permute_and_save directly).

If `CHUNKED = False`: 
3D single-pass: all data is held in memory at once.
Only suitable for small sample counts (smoke tests, etc.).
This can be used for smoke tests on Euler or for workstation calculations (where I have 512GB RAM)


`LOG_WANDB`
Set LOG_WANDB=1 in the environment to log this run to wandb (project sampler_3DRC). 

`SAMPLING_TYPE`
- uniform_3D_grouped
- lhs
- log
- combined_lhs_uniform
- combined_log_uniform
- combined_log_lhs

`CHUNK_SIZE`
Chunk size for the chunked sampling loop.
Each chunk requires ~240 MB (eps_g) + ~1.44 GB (dh) on CPU RAM.
Increase if you want fewer HDF5 append calls; decrease if GPU VRAM is tight.


`SAMPLE_2D`
Only sampling values for eps_g. Does not continue to calculate stresses. Depreciated version of sampler. 
2D path: sample membrane strains, permute top/bottom pairs → generalised
strains, and save eps_g only. Stress and stiffness are not computed here.

`REMOVE_OUTLIERS`
removes roughly 0.02% of the data containing values where D is extremely large. Can tune the amount with the "factor" given in the filter_outliers function