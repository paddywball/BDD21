#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 16:32:33 2020

@author: patrickball

to run: mpirun -np 8 python Fluidity_Optimize_Parallel.py
"""

import h5py
import re
import os
import vtk
import numpy as np
import pandas as pd
from glob import glob
from mpi4py import MPI

###~~~~~~~~~~~~~~~~~~~~~~~~~~~###
"""
Import all necessary modules from Fluidity_Chem_Mods.py
Import all necessary dictionaries from Fluidity_Chem_dict.py
"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~###

# Import the package.
from Fluidity_Chem_Mods import *

# Element sizes
from Fluidity_Chem_dict import Element_Sizes as RI

# Element Valancy States
from Fluidity_Chem_dict import Valency as VAL

# Choose Preferred Mantle Composition
#from invmel_dict import PM_MS_1995 as Conc
# If You Want To Mix Two Mantle Compositions
from Fluidity_Chem_dict import PM_MO_1995 as Conc1
from Fluidity_Chem_dict import DM_SS_2004 as Conc2

# Choose Preferred Partition Coefficients
# Required if colc_D_bar_P_bar function is used
from Fluidity_Chem_dict import D_MO_1995 as Part


###~~~~~~~~~~~~~~~~~~~~~~~~~~~###
"""
Run in parallel on multiple processors
"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~###

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
max_cpu = comm.Get_size()


###~~~~~~~~~~~~~~~~~~~~~~~~~~~###
"""
Define input parameters for individual Fluidity model.
"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~###

# Root to relevant directory for Fluidity model
os.chdir('Edge_80km_noflow')

# First time step to include
min_num_steps = 0

# Last time step to include +1
max_num_steps = 51

# Include potential temperature of system in oC
Tp = 1598. - 273.

# Length of the Y axis in km
Max_Y_Axis = 1000000.

# Gravity (m/s^2)
grav = 9.81

# Density of Solid Rock (kg/m^3)
density = 3300.

# Set mantle composition using linear mix of primitive (eNd = 0) and
# depleted (eNd = 10) mantle.
eNd = 10.

# Set elements to calculate concentrations of
elements = ['La','Ce','Pr','Nd','Sm','Eu','Gd','Tb','Dy','Ho','Er','Tm','Yb','Lu','Na','Ti','Hf','Rb','Sr','Th','U','Pb','Nb','Zr','Y','Ta','Sc','V','Cr','K','P','Ba']

# Calculate H2O wt% in the mantle source
H2O = (((10 - eNd)/10.) * 0.00028) + ((eNd/10.) * 0.000154)

# Set Melt as identifier for class Fluidity_Chem
Melt = Fluidity_Chem(bulk_H2O=H2O)


###~~~~~~~~~~~~~~~~~~~~~~~~~~~###
"""
Necessary modules.
"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~###

def add_string(a):
    # Concatenate all elements in two columns in array together.
    # Add in trailing zeros to make unique identifier.   
    return float(str(int(a[0])).zfill(3) + str(int(a[1])).zfill(8))


###~~~~~~~~~~~~~~~~~~~~~~~~~~~###
"""
Work flow to calculate incompatible element concentrations from
particle positions at each time step within a Fluidity model.
"""
###~~~~~~~~~~~~~~~~~~~~~~~~~~~###

# Make empty arrays to hold particle information.
    
time_step_melt = np.array([]) # Timesteps
partid_melt = np.array([]) # Particle IDs
procid_melt = np.array([]) # Processor IDs
partx_melt = np.array([]) # X Coordinates
party_melt = np.array([]) # Y Coordinates
parttemp_melt = np.array([]) # Temperatures
partKatzF_melt = np.array([]) # Melt Fraction
partKatzFmax_melt = np.array([]) # Maximum Melt Fraction
partKatzFdiff_melt = np.array([]) # Difference in Melt Fraction Between Timesteps.
partKatzFrate_melt = np.array([]) # Melting Rate
partlatent_melt = np.array([]) # Latentent Heat Consumed At Timestep.

#----------------------------------
    
# Make output dictionary for desired data
    
final = {}
   
final["Time_Step"] = np.array([])
final["partid"] = np.array([])
final["X_coord"] = np.array([])
final["Y_coord"] = np.array([])
final["Temperature"] = np.array([])
final["Melt_Fraction"] = np.array([])    
final["Melt_Diff"] = np.array([])
final["Melting_Rate_Ma"] = np.array([])
final["Latent_T_Loss"] = np.array([])
final["Upwelling_Rate"] = np.array([])
final["Time_Ma"] = np.array([])
   
# Add a dictionary entry for each element
    
for i in elements:
   
    final[str(i)] = np.array([])

#----------------------------------

# Make array of time in Myrs since the model began that corresponds to each time step.

vtus = sorted([vtu_file for vtu_file in glob('*.pvtu') if 'Inflow' not in vtu_file],
	      key=lambda value: int(re.compile(r'(\d+)').split(value)[-2]))

time_line = np.array([])

for path2vtu in vtus:
    vtkReader = vtk.vtkXMLPUnstructuredGridReader()
    vtkReader.SetFileName(path2vtu)
    vtkReader.Update()
    vtkOut = vtkReader.GetOutput()
    vtkScalars = vtkOut.GetPointData().GetScalars('EverythingExcept::Time')
    time = np.array(vtkScalars.GetTuple1(0))
    time_line = np.append(time_line, (time / 3.1536e+13))

#----------------------------------

# Open file containing particle data as h5f
    
with h5py.File('Pourquoi.particles.SubMelt.h5part', 'r') as h5f:

    # Find and append relevant particle data within the file into predefined arrays
    
    for i in range(min_num_steps, max_num_steps,1):
        partid_melt = np.append(partid_melt, h5f['Step#' + str(i)]['id'][()])
        procid_melt = np.append(procid_melt, h5f['Step#' + str(i)]['proc_id'][()]) 
        partx_melt = np.append(partx_melt, h5f['Step#' + str(i)]['x'][()])
        party_melt = np.append(party_melt, h5f['Step#' + str(i)]['y'][()])
        parttemp_melt = np.append(parttemp_melt, h5f['Step#' + str(i)]['Katz5'][()])          
        partKatzF_melt = np.append(partKatzF_melt, h5f['Step#' + str(i)]['Katz1'][()])   
        partKatzFmax_melt = np.append(partKatzFmax_melt, h5f['Step#' + str(i)]['Katz2'][()])
        partKatzFdiff_melt = np.append(partKatzFdiff_melt, h5f['Step#' + str(i)]['Katz3'][()])    
        partKatzFrate_melt = np.append(partKatzFrate_melt, h5f['Step#' + str(i)]['Katz3'][()])
        partlatent_melt = np.append(partlatent_melt, h5f['Step#' + str(i)]['Katz6'][()])
        time_step_melt = np.append(time_step_melt, ((h5f['Step#' + str(i)]['Katz1'][()] * 0.) + i ))

    # Combine particle ID and processor ID together to make a unique identifier for each particle.
    # IDs are followed by a series of zeros to ensure same structure for each particle and no acciendental
    # repeats.
                                                                                                                                                               
    totalid = np.array(list(zip(procid_melt, partid_melt)))
    totalid_melt = np.apply_along_axis(add_string, axis=1, arr=totalid)
        
    #----------------------------------
    
    # loop through each processor ID used to generate Fluidity model.
    for i in range(1,29,1):
   
        # Check if particles were calculated on processor i.
        time_step_proc = time_step_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
           
        if (len(time_step_proc) > 0):
            
            # Filter to only include particles where melting has occured by checking melt fraction > 0 at last timestep.
            id_proc_1 = totalid_melt[(partKatzFmax_melt > 0.) & (time_step_melt == (max_num_steps - 1)) & (totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            # Filter to only include paticles and time steps where melting occurred during time steps of interest
            id_proc_2 = totalid_melt[(partKatzFdiff_melt > 0.) & (totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            id_proc = np.intersect1d(id_proc_1, id_proc_2)

            # Filter relevant data for processor i
            partx_proc = partx_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            party_proc = party_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            parttemp_proc = parttemp_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1)*100000000.))]
            partKatzF_proc = partKatzF_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            partKatzFdiff_proc = partKatzFdiff_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            partKatzFrate_proc = partKatzFrate_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            partKatzFmax_proc = partKatzFmax_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            partlatent_proc = partlatent_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]
            totalid_proc = totalid_melt[(totalid_melt > (i*100000000.)) & (totalid_melt < ((i+1.)*100000000.))]

            #----------------------------------
            
            # Loop through each particle on processor and calculate melt composition
       
            j_max = int(len(id_proc))
            print(i, j_max)

            for j in range(0,j_max,1):
                # Run on multiple processors
                if j % (max_cpu) == rank:
                    
                    # Find all data associated with individual particle 
                    ids = id_proc[j]
                    time_step = time_step_proc[(totalid_proc == ids)]
                    partx = partx_proc[(totalid_proc == ids)] / 1000.
                    party = ((Max_Y_Axis - party_proc[(totalid_proc == ids)]) * grav * density) / 1e9
                    parttemp = parttemp_proc[(totalid_proc == ids)]
                    partKatzF = partKatzF_proc[(totalid_proc == ids)]
                    partKatzFdiff = partKatzFdiff_proc[(totalid_proc == ids)]
                    partKatzFrate = partKatzFrate_proc[(totalid_proc == ids)] * 3.1536e+13
                    partKatzFmax = partKatzFmax_proc[(totalid_proc == ids)]
                    partlatent = partlatent_proc[(totalid_proc == ids)]
                    partid = totalid_proc[(totalid_proc == ids)]

                    # Make array of time through model for particle

                    parttime = np.empty(np.size(time_step))
                    for i, Xi in enumerate(time_step):
                        parttime[i] = time_line[int(time_step[i])]

                    # Make array of upwelling velocity in cm / yr for particle
                    
                    partupwelling = np.empty(np.size(party))
                    partupwelling[0] = 0.                    
                    for i, Xi in enumerate(party[1:]):
                        partupwelling[i] = (((party[i] - party[i-1]) * 1e11 / (grav * density)) * -1.) / ((time_line[int(time_step[i])] - time_line[int(time_step[i-1])]) * 1e6)                      
                    
                    #----------------------------------
                    
                    # If particle begins below the solidus only include time steps at or above the solidus.

                    if partKatzFmax[0] == 0.:
                        # Only include time steps above the solidus
                        solidus = next((i for i, x in enumerate(partKatzFmax) if x!= 0.), None) - 1 # x!= 0 for strict match
               
                        time_step = time_step[solidus:]
                        partid = partid[solidus:]
                        partx = partx[solidus:]
                        party = party[solidus:]
                        parttemp = parttemp[solidus:]
                        partKatzF = partKatzF[solidus:]
                        partKatzFdiff = partKatzFdiff[solidus:]
                        partKatzFrate = partKatzFrate[solidus:] 
                        partlatent = partlatent[solidus:]
                        partupwelling = partupwelling[solidus:]
                        parttime = parttime[solidus:]

                    # Only include steps where melt fraction is increasing
                    
                    my_flgs = partKatzFdiff > 0.
                    my_flgs[0] = True                        
                    time_step = time_step[my_flgs]
                    partid = partid[my_flgs]
                    partx = partx[my_flgs]
                    party = party[my_flgs]
                    parttemp = parttemp[my_flgs]
                    partKatzF = partKatzF[my_flgs]
                    partKatzFrate = partKatzFrate[my_flgs]
                    partlatent = partlatent[my_flgs]
                    partupwelling = partupwelling[my_flgs]
                    parttime = parttime[my_flgs]
                    partKatzFdiff = partKatzFdiff[my_flgs]

                    # If maximum melt fraction is > 0.1% calculate melt composiion

                    if ((np.max(partKatzF) > 0.001)):
    
                        # Melt Composition Calculation for Each Particle
               
                        el_cl = Melt.Ball_Fluidity(Tp, eNd, Conc1, Conc2, Part, VAL, RI, elements, party, parttemp, partKatzF)
           
                        # Update final dictionary
                           
                        final["Time_Step"] = np.append(final["Time_Step"], time_step)
                        final["partid"] = np.append(final["partid"], partid)
                        final["X_coord"] = np.append(final["X_coord"], partx)
                        final["Y_coord"] = np.append(final["Y_coord"], party)
                        final["Temperature"] = np.append(final["Temperature"], parttemp)
                        final["Melt_Fraction"] = np.append(final["Melt_Fraction"], partKatzF)
                        final["Melt_Diff"] = np.append(final["Melt_Diff"], partKatzFdiff)
                        final["Melting_Rate_Ma"] = np.append(final["Melting_Rate_Ma"], partKatzFrate)
                        final["Latent_T_Loss"] = np.append(final["Latent_T_Loss"], partlatent)
                        final["Upwelling_Rate"] = np.append(final["Upwelling_Rate"], partupwelling) 
                        final["Time_Ma"] = np.append(final["Time_Ma"], parttime) 
   
                        for i in elements:

                            final[str(i)] = np.append(final[str(i)], el_cl[str(i)])

#----------------------------------

# Save dictionary of particle data to an output file per processer used to run script.
       
for ii in range(max_cpu): 
    if rank == ii:
        df = pd.DataFrame.from_records(final)
        df.to_csv(str('Full_model%i' %(rank) + '.csv'))

