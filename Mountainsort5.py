# -*- coding: utf-8 -*-
"""
Created on Tue May 27 11:43:28 2025

@author: Admin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  9 17:37:31 2025

@authors:leonardo.dellamea; francesco.mainardi
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

#%%
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



# ------------------------ SAVING RECORDING IN BATCHES ------------------------


# variables
fileDirectory = r"C:\Users\Admin\Desktop\Leonardo\Prova LOW"   #put the path
fileName  = r'\1481_DIV56_lowDens600_nbasal_001_00.brw' #filevname 
wellID = 'Well_A1' #change well 
basedir = r'C:\Users\Admin\Desktop\LOW DATA'
file_path_row= Path(str(fileDirectory + fileName))

useSyntheticNoise = True

total_sec = 500
Batch_size =100

n_b = int(total_sec/Batch_size)

for j in range(2,n_b) :
    dataStartPositionSec = j*Batch_size
    dataDurationSec = Batch_size
    
    print('Start sec',dataStartPositionSec)
    print('Finish sec',dataStartPositionSec + dataDurationSec)
    
    
    
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
    
    with h5py.File(file_path_row, 'r') as file:
    
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
    data_matrix = np.array((data_matrix*ConversionFactor) + OffsetValue,dtype=np.float32)
    # for s in range(data_matrix.shape[0]):
        
    #     data_matrix[s,:] = np.array((data_matrix[s,:]*ConversionFactor) + OffsetValue,dtype=np.float32)
    
    '''
    Construct the RASTER matrix used in Spike interface.
    
    '''
    file_path= basedir + f'\Data{j}.bin'
    recording = se.NumpyRecording(data_matrix.T, samplingRate)
    si.write_binary_recording(recording, file_path, dtype=np.float32)
    Number_electrodes =  recording.get_num_channels()
    # file_path_bin = file_path.parent.joinpath("Data_matrix.bin")
    # print(f"Traces dtype: {recording.get_dtype()}")
    
    # # Confirm that the data was loaded correctly by comparing the data shapes and see they match the MATLAB data
    # # print(recording.get_num_frames(), recording.get_num_channels())
    # print(recording.get_num_frames(), recording.get_num_channels())
    
    %matplotlib
    # PLOT TRACES
    timeseries_plot = sw.plot_traces(recording,channel_ids = np.arange(3,10) ,time_range=(1,1.2),figsize=(8,10))
    plt.title('Saved time series')
    plt.show()
  
    del data_matrix
    del recording




#%%
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
from spikeinterface import append_recordings,concatenate_recordings
## KILOSORT

from kilosort import io
from kilosort import run_kilosort
from kilosort.io import save_probe
from kilosort.io import load_probe



# variables
fileDirectory = r"C:\Users\Admin\Desktop\Leonardo\Prova LOW"   #put the path
fileName  = r'\1481_DIV56_lowDens600_nbasal_001_00.brw' #filevname 
wellID = 'Well_A1' #change well 
basedir = r'C:\Users\Admin\Desktop\LOW DATA'
file_path_row= Path(str(fileDirectory + fileName))
Number_electrodes = 2304
with h5py.File(file_path_row, 'r') as file:

    # collect experiment information
    
    samplingRate = file.attrs['SamplingRate']
    chIdxs = np.array(file[wellID + '/StoredChIdxs'])

    
    # Variables for covertion from digital to analog
    
    minDigitalValue = file.attrs['MinDigitalValue']
    maxDigitalValue = file.attrs['MaxDigitalValue'] 
    minAnalogValue = file.attrs['MinAnalogValue'] 
    maxAnalogValue = file.attrs['MaxAnalogValue'] 
    
    ConversionFactor =  (maxAnalogValue - minAnalogValue) / (maxDigitalValue - minDigitalValue) 
    OffsetValue =    minAnalogValue - ConversionFactor * minDigitalValue

# ------------------------ RECONSTRUCTING THE WHOLE SIGNAL ------------------------
file_path_bin = r"C:\Users\Admin\Desktop\LOW DATA\Data0.bin"
recc1 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                   num_channels=Number_electrodes)

file_path_bin = r"C:\Users\Admin\Desktop\LOW DATA\Data1.bin"
recc2 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                   num_channels=Number_electrodes)
file_path_bin = r"C:\Users\Admin\Desktop\LOW DATA\Data2.bin"
recc3 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                   num_channels=Number_electrodes)



recording_list = [recc1, recc2,recc3]
full_rec = concatenate_recordings(recording_list)
print(full_rec)
timeseries_plot = sw.plot_traces(full_rec,channel_ids = np.arange(3,20) ,time_range=(1,1.2))
#%%
#------------- Probe -------------
# The electrodes are set in square geometry
  
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

k = 0
n = int(math.sqrt(Number_electrodes))
positions = np.zeros((Number_electrodes, 2))
for i in range(n):
    for j in range(n):
        x = i * pitch
        y = j * pitch
        positions[k] = x, y
        k =k+1
        

# Generate the probe variable

probe_2d = Probe(ndim=2, si_units='um')
probe_2d.set_contacts(positions=positions, shapes='square', shape_params={'width': Electrode_size})

# Set channel indicies
Ch_idx = np.arange(Number_electrodes)
probe_2d.set_device_channel_indices(Ch_idx)

# Plot the probe
plot_probe(probe_2d)

#------------- Filter the data -------------

timeseries_plot = sw.plot_traces(full_rec,channel_ids = np.arange(3,10) ,time_range=(1,1.2))

plt.title('RAW')
plt.show()


recording_CAR = spre.common_reference(full_rec,operator = 'average')


timeseries_plot = sw.plot_traces(recording_CAR,channel_ids = np.arange(3,10) ,time_range=(1,1.2))
plt.title('CAR')

plt.show()


recording_BUT = spre.highpass_filter(recording_CAR, freq_min=300.0, margin_ms=2.0)

timeseries_plot = sw.plot_traces(recording_BUT,channel_ids = np.arange(3,10) ,time_range=(1,1.2))
plt.title('BUTTERWORTH')

plt.show()

## Use local whitening, given that the pitch between electrodes is 60 um set the radius at 240 um that correspond to 4 ch
recording_WT = spre.whiten(recording_BUT.set_probe(probe_2d), mode = 'local',radius_um=240.0)


timeseries_plot = sw.plot_traces(recording_WT,channel_ids = np.arange(3,10) ,time_range=(1,1.2))
plt.title('WHITENED')

plt.show()


dataDurationSec = recording_WT.get_duration()


#------------- Mountainsort5 parameters -------------



####### SCHEME 2 ALGORITHM: refer to https://github.com/flatironinstitute/mountainsort5/blob/main/docs/scheme2.md

# Phase 1. First, a subset of the recording is extracted for training, according to the training_duration_sec and training_recording_sampling_mode 
# parameters. Then, scheme 1 detection and clustering is performed on this training subset.

# Phase 2. In this phase, a series of classifiers, one for each channel, are trained using the snippets and labels obtained from phase 1.
# The procedure used here is difficult to describe, so for now I will refer the reader to the documented source code in sorting_scheme2.py.

# Phase 3. Once the classifiers have been fit, new events are detected throughout the entire recording using the detect_threshold and 
# detect_channel_radius parameters (possibly different from phase1_detect_threshold and phase1_detect_channel_radius). In order to detect and 
# sort more events that overlap in time, it is desirable to choose detect_channel_radius to be smaller than phase1_detect_channel_radius, so that events 
# not included in phase 1 can be included in this final phase. The classifiers trained in phase 2 are then used to label each event. Again, for details
# we refer to the reader to the documented source code in sorting_scheme2.py.


sorting_params= ss.get_default_sorter_params('mountainsort5')

sorting_params["scheme"] = '2'



# Amount of data (in seconds) loaded into memory at a time during the extraction / classification step of scheme 2
chunk = 60
sorting_params["chunk_duration"] = str(chunk)+'s'




# The detect_sign parameter determines whether to detect positive-going or negative-going 
# spikes. The default value is -1, which is appropriate for most recordings. The value 1 is appropriate for 
# recordings in which the spikes are positive-going. You can also use a value of 0 to detect both positive and negative-going spikes.

sorting_params["detect_sign"] = 0

#################


# The detection threshold (detect_threshold) is one of the more difficult parameters to optimize. If it is set too low, 
# then you may end up with false merges because clusters will overlap for spikes with lower amplitudes. This is especially 
# relevant for bursting cells that exhibit a range of spike amplitudes. On the other hand, if this parameter is set too high, 
# then you'll miss some low-amplitude events. I would recommend leaving this at the default value (5.5 for whitened data) since this 
# seems to perform well on a variety of examples I have tried. You may think that 5.5 sounds high, but note that detection is performed 
# on the ZCA-whitened data for which the spike peaks tend to be better separated from noise compared with the pre-whitened bandpass-filtered data.

sorting_params["detect_threshold"] = 4.5

##################

# This parameter determines the minimum allowable time interval between detected spikes in the same spatial region (see detect_channel_radius). 
# This is useful for avoiding double-detection of individual spikes. The default value is 0.5 msec for scheme 1. The first phase of scheme 2 uses scheme 1,
# and in that setting the default value is 1.5 msec in order to avoid including overlapping spikes in the intitial clustering pass.
 
sorting_params["detect_time_radius_msec"] = 1.5  

##################

# This parameter is a companion to detect_time_radius_msec. It determines the spatial radius (in units of the channel locations) 
# within which the detection time radius applies. In general, a spike is detected at timepoint t on channel m if it is the largest amplitude 
# event within a time window size detect_time_radius_msec around t and within a spatial neighborhood around channel m defined by detect_channel_radius. 
# The default is None which effectively gives an infinite spatial radius.
# This is just like the phase1_detect_channel_radius parameter, but it is used for phase 2.

sorting_params["scheme2_detect_channel_radius"] = 50

#################

#sorting_params["scheme2_max_num_snippets_per_training_batch"] = 200

# The detect_channel_radius parameter for phase 1

sorting_params["scheme2_phase1_detect_channel_radius"] = 50  ## I don't want more channels condiered during detection. These are not close enought to 
                                                             ## a neuron to be detected simultanoeusly by more channels.

#################


# This parameter affects dimension reduction prior to clustering. If the dataset has M channels, then M * npca_per_channel PCA features are
# computed for each spike snippet. The default value is 3, which is appropriate for most recordings.

sorting_params["npca_per_channel"] = 4

#################

# MountainSort utilizes a subdivision method for clustering. After extracting spike snippets from the preprocessed traces, the data undergo 
# dimension reduction through PCA before clustering. Initially, npca_per_subdivision PCA features are computed, followed by Isosplit clustering.
# If more than one cluster is detected, the dataset is split into two subdivisions, and feature extraction is performed again within each subdivision
# separately, followed by Isosplit clustering to subdivide the clusters further. This process is repeated recursively until each leaf subdivision
# returns a single cluster. At each stage, the same number of PCA components (npca_per_subdivision) is used. Recomputing features on each subdivision
# offers an advantage, as it allows the refined components to capture features that can better differentiate between clusters within the same overall
# feature space region.

sorting_params["npca_per_subdivision"] =10

#################



# The duration of the training data (in seconds). See also the training_recording_sampling_mode parameter.

sorting_params["scheme2_training_duration_sec"] = int(dataDurationSec/4) #1/4 of the recording is used as training data

#################


# How to sample the training data. If 'initial', then the first training_duration_sec of the recording will be used. If 'uniform', then the training 
# data will be sampled uniformly in 10-second chunks from the recording.

sorting_params["scheme2_training_recording_sampling_mode"] = 'uniform'

#################


# These parameters determine the time duration (units of samples) of the extracted spike snippets. The total duration of the snippet is 
# snippet_T1 + snippet_T2 samples, with snippet_T1 samples before the peak and snippet_T2 samples after the peak.

sorting_params["snippet_T1"] = int(0.8*(samplingRate/1000))
sorting_params["snippet_T2"] = int(1.5*(samplingRate/1000))

#################


# If specified, this parameter determines the spatial radius (in units of the channel locations) of a mask that is applied to the extracted spike
# snippets. For spikes centered at channel m the mask includes all channels within snippet_mask_radius of channel m, and the signal outside of the
# mask region is set to zero. This is useful for removing signal from time-overlapping, spatially-separated events during the clustering phase.

sorting_params["snippet_mask_radius"] = 10

#################

sorting_params["whiten"] = False
sorting_params["filter"] = False
sorting_params["delete_temporary_recording"] = True



#%
#------------- Run sorter -------------


#The first two channels are of calibration...
bad_ch = [0,1]
recording_WT = recording_WT.remove_channels(bad_ch)

sorting_MS5 = ss.run_sorter(
    sorter_name='mountainsort5', recording=recording_WT, output_folder="Mountainsort5_output",**sorting_params,remove_existing_folder = True,
)



print(sorting_MS5)

#%
#------------- POST PROCESSING -------------
# Create analyszer


# Generate the probe variable

probe_2d = Probe(ndim=2, si_units='um')
probe_2d.set_contacts(positions=positions[2:,:], shapes='square', shape_params={'width': Electrode_size})

# Set channel indicies
Ch_idx = np.arange(Number_electrodes-2)
probe_2d.set_device_channel_indices(Ch_idx)

# Plot the probe
plot_probe(probe_2d)


analyzer = si.create_sorting_analyzer(sorting=sorting_MS5 , recording=recording_WT.set_probe(probe_2d), format='binary_folder', folder='aanalyzer_binary',overwrite=True)

# core extensions
analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels","unit_locations", "spike_locations", "template_metrics"])
analyzer.compute('principal_components', n_components = 5, mode="by_channel_local")
# more extensions
analyzer.compute(['spike_amplitudes', 'correlograms', 'template_similarity', 'quality_metrics'])
  
#%                      
metrics = sqm.compute_quality_metrics(analyzer, metric_names=["snr", "isi_violation", "nearest_neighbor"])


#%%
unit_ids = sorting_MS5.unit_ids[:Number_electrodes]



#%
# PLOT SOME RESULTS
sw.plot_all_amplitudes_distributions(analyzer, figsize=(10, 10))
plt.show()

# oN THE PROBE

sw.plot_unit_locations(analyzer, figsize=(4, 8))

plt.show()


sw.plot_rasters(analyzer)
plt.show()







