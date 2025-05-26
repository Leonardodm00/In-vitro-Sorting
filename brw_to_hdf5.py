# -*- coding: utf-8 -*-
"""
Created on Mon May 26 14:54:46 2025

@author: Admin
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import torch
## SPIKE INTERFACE MOULE AND SUBMODULES
import spikeinterface as si
from pathlib import Path
from pprint import pprint
import math
import spikeinterface as si  # import core only
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw
import scipy.io

## PROBE

from probeinterface import Probe
from probeinterface.plotting import plot_probe
from probeinterface import write_prb, read_prb
from probeinterface import ProbeGroup
## KILOSORT

from kilosort import io
from kilosort import run_kilosort
from kilosort.io import save_probe
from kilosort.io import load_probe



#%
def create_directory(dir_path):
    """
    Creates a directory at the specified path.

    Args:
        dir_path (str): The path where the directory should be created.
    """
    try:
        # Check if the directory already exists
        if os.path.exists(dir_path):
            print(f"Directory '{dir_path}' already exists.")
        else:
            # Create the directory
            os.makedirs(dir_path)
            print(f"Directory '{dir_path}' created successfully.")
    except OSError as e:
        print(f"Error creating directory '{dir_path}': {e}")



def DecodeEventBasedRawData(file, data, wellID, startFrame, numFrames):
    # collect the TOCs
    toc = np.array(file['TOC'])
    eventsToc = np.array(file[wellID + '/EventsBasedSparseRawTOC'])
    # from the given start position and duration in frames, localize the corresponding event positions
    # using the TOC
    tocStartIdx = np.searchsorted(toc[:, 1], startFrame)
    tocEndIdx = min(np.searchsorted(toc[:, 1], startFrame + numFrames, side='right')
     + 1, len(toc) - 1)
    eventsStartPosition = eventsToc[tocStartIdx]
    eventsEndPosition = eventsToc[tocEndIdx]
    # decode all data for the given well ID and time interval
    binaryData = file[wellID + '/EventsBasedSparseRaw'][eventsStartPosition:eventsEndPosition]
    binaryDataLength = len(binaryData)
    pos = 0
    while pos < binaryDataLength:
        chIdx = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        chDataLength = int.from_bytes(binaryData[pos:pos + 4], byteorder='little', signed=True)
        pos += 4
        chDataPos = pos
        while pos < chDataPos + chDataLength:
            fromInclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
            pos += 8
            toExclusive = int.from_bytes(binaryData[pos:pos + 8], byteorder='little', signed=True)
            pos += 8
            rangeDataPos = pos
            for j in range(fromInclusive, toExclusive):
                if j >= startFrame + numFrames:
                    break
                if j >= startFrame:
                    data[chIdx][j - startFrame] = int.from_bytes(
                        binaryData[rangeDataPos:rangeDataPos + 2], byteorder='little', signed=True)
                rangeDataPos += 2
            pos += (toExclusive - fromInclusive) * 2
            
            
            
            
            
def GenerateSyntheticNoise(file, data, wellID, startFrame, numFrames):
# collect the TOCs
    toc = np.array(file['TOC'])
    noiseToc = np.array(file[wellID + '/NoiseTOC'])
    # from the given start position in frames, localize the corresponding noise positions
     # using the TOC
    tocStartIdx = np.searchsorted(toc[:, 1], startFrame)
    noiseStartPosition = noiseToc[tocStartIdx]
    noiseEndPosition = noiseStartPosition
    for i in range(tocStartIdx + 1, len(noiseToc)):
        nextPosition = noiseToc[i]
        if nextPosition > noiseStartPosition:
            noiseEndPosition = nextPosition
            break
    if noiseEndPosition == noiseStartPosition:
        for i in range(tocStartIdx - 1, 0, -1):
            previousPosition = noiseToc[i]
            if previousPosition < noiseStartPosition:
                noiseEndPosition = noiseStartPosition
                noiseStartPosition = previousPosition
                break
    # obtain the noise info at the start position
    noiseChIdxs = file[wellID + '/NoiseChIdxs'][noiseStartPosition:noiseEndPosition]
    noiseMean = file[wellID + '/NoiseMean'][noiseStartPosition:noiseEndPosition]
    noiseStdDev = file[wellID + '/NoiseStdDev'][noiseStartPosition:noiseEndPosition]
    noiseLength = noiseEndPosition - noiseStartPosition
    noiseInfo = {}
    meanCollection = []
    stdDevCollection = []
    for i in range(1, noiseLength):
        noiseInfo[noiseChIdxs[i]] = [noiseMean[i], noiseStdDev[i]]
        meanCollection.append(noiseMean[i])
        stdDevCollection.append(noiseStdDev[i])
    # calculate the median mean and standard deviation of all channels to be used for
    # invalid channels
    dataMean = np.median(meanCollection)
    dataStdDev = np.median(stdDevCollection)
# fill with Gaussian noise
    for chIdx in data:
        if chIdx in noiseInfo:
            data[chIdx] = np.array(np.random.normal(noiseInfo[chIdx][0], noiseInfo[chIdx][1],
                                                    numFrames), dtype=np.int16)
        else:
            data[chIdx] = np.array(np.random.normal(dataMean, dataStdDev, numFrames),
                                   dtype=np.int16)

# variables
fileDirectory = r"C:\Users\Admin\Desktop\Leonardo\Prova LOW"   #put the path
Probe_name = 'CorePlate_6W_38_60.json'
fileName  = r'\1481_DIV56_lowDens600_nbasal_001_00.brw' #filevname 

Probe_path = r"C:\Users\Admin\Desktop\Leonardo\Prova LOW\CorePlate_6W_38_60.json"
wellID = 'Well_A1' #change well 
file_path= Path(str(fileDirectory + fileName))
print(file_path)

useSyntheticNoise = True
dataStartPositionSec = 0
dataDurationSec = 10
'''
First 2 channels are used for calibration... Disregard them



'''

'''
Probe info

'''

# # CorePlate™ 1W 38/60
# pitch = 60 # um   
# Electrode_size = 21 # um


# # CorePlate™ 1W 27/42L
# pitch = 42 # um   
# Electrode_size = 21 # um


# # CorePlate™ 1W 51/81
# pitch = 81 # um   
# Electrode_size = 21 # um

# CorePlate™ 6W 38/60
pitch = 60 # um   
Electrode_size = 25 # um





'''
Extraction

'''

with h5py.File(file_path, 'r') as file:

    # collect experiment information
    
    samplingRate = file.attrs['SamplingRate']
    chIdxs = np.array(file[wellID + '/StoredChIdxs'])
    
    # convert the requested time interval from seconds to frames
    startFrame = int(dataStartPositionSec * samplingRate)
    numFrames = int(dataDurationSec * samplingRate)
    
    
    # Variables for covertion from digital to analog
    
    minDigitalValue = file.attrs['MinDigitalValue']
    maxDigitalValue = file.attrs['MaxDigitalValue'] 
    minAnalogValue = file.attrs['MinAnalogValue'] 
    maxAnalogValue = file.attrs['MaxAnalogValue'] 
    
    ConversionFactor =  (maxAnalogValue - minAnalogValue) / (maxDigitalValue - minDigitalValue) 
    OffsetValue =    minAnalogValue - ConversionFactor * minDigitalValue 
    
    
    
    
    # initialize an empty (fill with zeros) data collection
    data = {chIdx: np.zeros(numFrames, dtype=np.int16) for chIdx in chIdxs} 
    # fill the data collection with Gaussian noise if requested
    if useSyntheticNoise:
            GenerateSyntheticNoise(file, data, wellID, startFrame, numFrames)
    # fill the data collection with the decoded event based sparse raw data
    DecodeEventBasedRawData(file, data, wellID, startFrame, numFrames)


data_matrix = np.array(list(data.values()), dtype=np.int16)





#%%

num_channels = 2304

channel_recordings = []
for i in range(2,num_channels):

    channel_recordings.append(data_matrix[i,:])
del data_matrix

#%%
# --- Define the HDF5 file path ---
hdf5_filename = 'Multi_Channel_Data.hdf5'
os.chdir(r'C:\Users\Admin\Desktop\LOW DATA')

# Clean up previous file if it exists for clean testing
if os.path.exists(hdf5_filename):
    os.remove(hdf5_filename)
    print(f"Removed existing file: {hdf5_filename}")

# --- Create the HDF5 file and structure ---
with h5py.File(hdf5_filename, 'w') as f:
    # 1. Create the 'parameters' group
    parameters_group = f.create_group('parameters')

    # Store parameters as attributes within the 'parameters' group
    # Attributes are good for small pieces of metadata associated with a group or dataset.
    parameters_group.attrs['sampling_frequency'] = samplingRate
    parameters_group.attrs['offset_value'] = OffsetValue
    parameters_group.attrs['conversion_factor'] = ConversionFactor
    print(f"Created group '/parameters' and added attributes.")

    # Alternatively, store parameters as datasets (more flexible if they were arrays)
    # parameters_group.create_dataset('sampling_frequency', data=sampling_frequency)
    # parameters_group.create_dataset('offset_value', data=offset_value)
    # parameters_group.create_dataset('conversion_factor', data=conversion_factor)


    # 2. Create the 'recordings' group
    recordings_group = f.create_group('recordings')
    print(f"Created group '/recordings'.")

    # 3. Create sub-groups for each channel and store data
    for i, recording_data in enumerate(channel_recordings):
        channel_name = f'channel_{i:02d}' # Format as 'channel_00', 'channel_01', etc.
        channel_group = recordings_group.create_group(channel_name)

        # Create a dataset within each channel group
        # 'data' is the name of the dataset storing the actual recording
        # Using compression ('gzip') and chunking can be very beneficial for large datasets
        # chunking=True lets h5py decide optimal chunks, or you can specify a tuple like (1000,)
        channel_group.create_dataset('data', data=recording_data, dtype=np.int16, chunks=True)
        print(f"  Created group '/recordings/{channel_name}' and dataset 'data'.")

print(f"\nSuccessfully created HDF5 file: {hdf5_filename}")
