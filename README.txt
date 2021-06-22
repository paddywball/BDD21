###~~~~~~~~~~~~~~~~~~###
Fluidity Chemistry Model
###~~~~~~~~~~~~~~~~~~###

Fluidity Chemistry Model is a series of Python scripts which
can be used together to estimate melt chemistry within a Fluidity model.

For a full explanation of this modelling software please see Ball et al.,
(submitted to JGR Solid Earth).

#--------------------

# Software Requirements

python 2.7
numpy 1.16.5
h5py 2.9.0
re 2.2.1
pandas 0.24.2
scipy 1.2.1
os
vtk
glob
mpi4py

#--------------------

# Data Requirements

A Fluidity model of mantle convection within which X coordinate,
Y coordinate, Temperature, Melt Fraction, Melting Rate, Latent Heat
of Melting, Particle ID and Processor ID recorded for each particle
at each time step in a h5 file.

vtu or pvtu files of each time step within the model

#--------------------

# Example data files

We include example data files for an edge-driven-convection model
with a original oceanic thermal boundary layer thickness of 80 km.

# Files used to generate fluidity model
Constants.py
Par_Pourquoi.flml
Pourquoi.flml

# h5 file with particle data
Pourquoi.particles.SubMelt.h5part

# vtu/pvtu files
Pourquoi_*.pvtu
Pourquoi_*/Pourquoi_*_*.vtu

#--------------------

# Implementation

Place following files in the same directory:

# Script to calculate melt compositions from particle data
Fluidity_Optimize_Parallel_Example.py
# File including all the modules used in Fluidity_Optimize_Parallel_Example.py
Fluidity_Chem_Mods.py
# File including all datasets used in Fluidity_Optimize_Parallel_Example.py
Fluidity_Chem_dict.py 

Place Fluidity model data files in the same directory as each other, not 
necessarilty the same directory as the data processing scripts.

Edit lines 65-96 in Fluidity_Optimize_Parallel_Example.py to match the 
specific Fluidity model.

Edit line 173 in Fluidity_Optimize_Parallel_Example.py to match the 
specific Fluidity model.

Edit line 209 in Fluidity_Optimize_Parallel_Example.py to match the
specific Fluidity model.

To run the model, migrate to directory holding the python files and input the
following command:

mpirun -np 8 python Fluidity_Optimize_Parallel.py

where 8 is changed to the number of processors that can be used.

#--------------------

# Post Implementation

Model yields one csv file for each processor used to run 
Fluidity_Optimize_Parallel.py

Headers (i.e., first row) for all but one of the csv files should be removed
and all csv files should be appended beneath the file with a header remaining.

#--------------------

# Example Final Model

Full_model.csv
