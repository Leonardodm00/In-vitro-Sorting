


import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
import torch
from pathlib import Path
from pprint import pprint
import math
import scipy.io
from tempfile import TemporaryDirectory
import tempfile
import shutil

## SPIKE INTERFACE MODULE AND SUBMODULES
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm
import spikeinterface.comparison as sc
import spikeinterface.exporters as sexp
import spikeinterface.curation as scur
import spikeinterface.widgets as sw

## PROBE
from probeinterface import Probe
from probeinterface.plotting import plot_probe
from probeinterface import write_prb, read_prb
from probeinterface import ProbeGroup

## MOUNTAINSORT5
import mountainsort5 as ms5
from mountainsort5.util import create_cached_recording

'''
Notes:
    1) For now LUSSAC method is not used



'''
#%



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

# Clear the workspace
# %reset -f
# ------------------------ SAVING RECORDING IN BATCHES ------------------------

wellID_array = ['Well_A1'] #change well 
# 'Well_A1','Well_A2','Well_A3','Well_B1','Well_B2','Well_B3'


for wellID in wellID_array:


    # variables
    fileDirectory = r"C:\Users\l.collo\Desktop\Leo\DATA LOW\1263_DIV56_lowDens400_nbasal_secondaRecSenza bande_007_00"   #put the path
    fileName  = r'\1263_DIV56_lowDens400_nbasal_secondaRecSenza bande_007_00.brw' #filevname 
    
    # Create the base output directory with the wellID embedded
    basedir_out = fr'C:\Users\l.collo\Desktop\Leo\DATA LOW\SORTER OUTPUT_lowDens400{wellID}'
    
    # Check if the directory exists and create it if it doesn't
    if not os.path.exists(basedir_out):
        os.makedirs(basedir_out)
        print(f"Created directory: {basedir_out}")
    else:
        print(f"Directory already exists: {basedir_out}")
    
    
    
    file_path_row= Path(str(fileDirectory + fileName))
    
    useSyntheticNoise = True
    
    total_sec =600
    Batch_size =100
    
    n_b = int(total_sec/Batch_size)
    
    for j in range(0,n_b) :
        dataStartPositionSec = j*Batch_size
        dataDurationSec = Batch_size
        
        print('Start sec',dataStartPositionSec)
        print('Finish sec',dataStartPositionSec + dataDurationSec)
        
        
        
        '''
        First 2 channels are used for calibration... Disregard them
        
        
        
        '''
        
      
        
        
        
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
        file_path= basedir_out + f'\Data{j}.bin'
        recording = se.NumpyRecording(data_matrix.T, samplingRate)
        si.write_binary_recording(recording, file_path, dtype=np.float32)
        Number_electrodes =  recording.get_num_channels()
        # file_path_bin = file_path.parent.joinpath("Data_matrix.bin")
        # print(f"Traces dtype: {recording.get_dtype()}")
        
        # # Confirm that the data was loaded correctly by comparing the data shapes and see they match the MATLAB data
        # # print(recording.get_num_frames(), recording.get_num_channels())
        # print(recording.get_num_frames(), recording.get_num_channels())
        
        # %matplotlib
        # PLOT TRACES
        # timeseries_plot = sw.plot_traces(recording,channel_ids = np.arange(3,10) ,time_range=(1,1.2),figsize=(8,10))
        # plt.title('Saved time series')
        # plt.show()
      
        del data_matrix
        del recording
    
    
    
    
    
    # %reset -f
    
    # variables
    fileDirectory = r"C:\Users\l.collo\Desktop\Leo\DATA LOW\1263_DIV56_lowDens400_nbasal_secondaRecSenza bande_007_00"   #put the path
    fileName  = r'\1263_DIV56_lowDens400_nbasal_secondaRecSenza bande_007_00.brw' #filevname 
    
    # Create the base output directory with the wellID embedded
    basedir_out = fr'C:\Users\l.collo\Desktop\Leo\DATA LOW\SORTER OUTPUT_lowDens400{wellID}'
    
    # Check if the directory exists and create it if it doesn't
    if not os.path.exists(basedir_out):
        os.makedirs(basedir_out)
        print(f"Created directory: {basedir_out}")
    else:
        print(f"Directory already exists: {basedir_out}")
    
    
    
    file_path_row= Path(str(fileDirectory + fileName))
      
    # if __name__ == '__main__':
    
    # variables
    file_path_row= Path(str(fileDirectory + fileName))
    Number_electrodes = 2304
     
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
        
        
        
        
        
    
    # ------- SPLIT Channels -------
    
    split_amount = 4
    Full_probe_idx =np.arange(2304) # Create the initial array
       
    
    # Split the array into four equal parts
    Channel_Idx = np.split(Full_probe_idx, split_amount)
    
    Num_channels = int(Number_electrodes/split_amount)
    
    # Main variables
    
    summed_units = 0
    
    Raster_complete= []
    Position_complete = []
    
    # -------------------- MAIN LOOP ----------------------
    it = 0
    for set_ch in Channel_Idx:
        print('NEW ITERATION: {}'.format(it))
        print('Channel used:')
        print(set_ch)
        # # ------------------------ RECONSTRUCTING THE WHOLE SIGNAL ------------------------
        file_path_bin = fr"C:\Users\l.collo\Desktop\Leo\DATA LOW\SORTER OUTPUT_lowDens400{wellID}\Data0.bin"
        recc1 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                           num_channels=Number_electrodes)
        
        file_path_bin = fr"C:\Users\l.collo\Desktop\Leo\DATA LOW\SORTER OUTPUT_lowDens400{wellID}\Data1.bin"
        recc2 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                           num_channels=Number_electrodes)
        file_path_bin = fr"C:\Users\l.collo\Desktop\Leo\DATA LOW\SORTER OUTPUT_lowDens400{wellID}\Data2.bin"
        recc3 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                           num_channels=Number_electrodes)
        file_path_bin = fr"C:\Users\l.collo\Desktop\Leo\DATA LOW\SORTER OUTPUT_lowDens400{wellID}\Data3.bin"
        recc4 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                           num_channels=Number_electrodes)
        file_path_bin = fr"C:\Users\l.collo\Desktop\Leo\DATA LOW\SORTER OUTPUT_lowDens400{wellID}\Data4.bin"
        recc5 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                           num_channels=Number_electrodes)
        file_path_bin = fr"C:\Users\l.collo\Desktop\Leo\DATA LOW\SORTER OUTPUT_lowDens400{wellID}\Data5.bin"
        recc6 = se.BinaryRecordingExtractor(file_paths=file_path_bin, sampling_frequency=samplingRate,dtype=np.float32,
                                           num_channels=Number_electrodes)
        
        
        
        
        # recording_list = [recc1]
        recording_list = [recc1, recc2,recc3,recc4,recc5,recc6]
        full_rec = si.concatenate_recordings(recording_list)
        print(full_rec)
        full_rec = full_rec.select_channels(channel_ids=set_ch)
        print(full_rec)
        # timeseries_plot = sw.plot_traces(full_rec,channel_ids = np.arange(3,20) ,time_range=(1,1.2))
        del recc1
        del recc2
        del recc3
        del recc4
        del recc5
        del recc6
      
        Num_channels = len(set_ch)
        
        Probe_range =np.arange(Num_channels)
        
        
        
        # Generate the probe variable
        
        probe_2d = Probe(ndim=2, si_units='um')
        probe_2d.set_contacts(positions=positions[set_ch,:], shapes='square', shape_params={'width': Electrode_size})
        
        
        probe_2d.set_device_channel_indices(Probe_range)
    
        
       
        
       
        
        # Plot the probe
        # plot_probe(probe_2d)
        
        #------------- Filter the data -------------
        
        # timeseries_plot = sw.plot_traces(full_rec,channel_ids = np.arange(3,10) ,time_range=(1,1.2))
        
        # plt.title('RAW')
        # plt.show()
        
        
        recording_CAR = spre.common_reference(full_rec,operator = 'average')
        
        
        # timeseries_plot = sw.plot_traces(recording_CAR,channel_ids = np.arange(3,10) ,time_range=(1,1.2))
        # plt.title('CAR')
        
        # plt.show()
        
        
        recording_BUT = spre.highpass_filter(recording_CAR, freq_min=300.0, margin_ms=2.0)
        
        # timeseries_plot = sw.plot_traces(recording_BUT,channel_ids = np.arange(3,10) ,time_range=(1,1.2))
        # plt.title('BUTTERWORTH')
        
        # plt.show()
        
        ## Use local whitening, given that the pitch between electrodes is 60 um set the radius at 240 um that correspond to 4 ch
        recording_WT = spre.whiten(recording_BUT.set_probe(probe_2d), mode = 'local',radius_um=240.0)
        
        
        # timeseries_plot = sw.plot_traces(recording_WT,channel_ids = np.arange(3,10) ,time_range=(1,1.2))
        # plt.title('WHITENED')
        
        # plt.show()
        
        
        dataDurationSec = recording_WT.get_duration()
        
        
        
        #%
        
        
        
        
        
        
       
        # ------------- Mountainsort5 parameters -------------
        
        
        
        # ####### SCHEME 2 ALGORITHM: refer to https://github.com/flatironinstitute/mountainsort5/blob/main/docs/scheme2.md
        
        # # Phase 1. First, a subset of the recording is extracted for training, according to the training_duration_sec and training_recording_sampling_mode 
        # # parameters. Then, scheme 1 detection and clustering is performed on this training subset.
        
        # # Phase 2. In this phase, a series of classifiers, one for each channel, are trained using the snippets and labels obtained from phase 1.
        # # The procedure used here is difficult to describe, so for now I will refer the reader to the documented source code in sorting_scheme2.py.
        
        # # Phase 3. Once the classifiers have been fit, new events are detected throughout the entire recording using the detect_threshold and 
        # # detect_channel_radius parameters (possibly different from phase1_detect_threshold and phase1_detect_channel_radius). In order to detect and 
        # # sort more events that overlap in time, it is desirable to choose detect_channel_radius to be smaller than phase1_detect_channel_radius, so that events 
        # # not included in phase 1 can be included in this final phase. The classifiers trained in phase 2 are then used to label each event. Again, for details
        # # we refer to the reader to the documented source code in sorting_scheme2.py.
        
        
        # sorting_params= ss.get_default_sorter_params('mountainsort5')
        
        # sorting_params["scheme"] = '2'
        # sorting_params["n_jobs"] = 10
        
        # # Amount of data (in seconds) loaded into memory at a time during the extraction / classification step of scheme 2
        # chunk = 25
        # sorting_params["chunk_duration"] = str(chunk)+'s'
        
        
        
        
        # # The detect_sign parameter determines whether to detect positive-going or negative-going 
        # # spikes. The default value is -1, which is appropriate for most recordings. The value 1 is appropriate for 
        # # recordings in which the spikes are positive-going. You can also use a value of 0 to detect both positive and negative-going spikes.
        
        # sorting_params["detect_sign"] = 0
        
        # #################
        
        
        # # The detection threshold (detect_threshold) is one of the more difficult parameters to optimize. If it is set too low, 
        # # then you may end up with false merges because clusters will overlap for spikes with lower amplitudes. This is especially 
        # # relevant for bursting cells that exhibit a range of spike amplitudes. On the other hand, if this parameter is set too high, 
        # # then you'll miss some low-amplitude events. I would recommend leaving this at the default value (5.5 for whitened data) since this 
        # # seems to perform well on a variety of examples I have tried. You may think that 5.5 sounds high, but note that detection is performed 
        # # on the ZCA-whitened data for which the spike peaks tend to be better separated from noise compared with the pre-whitened bandpass-filtered data.
        
        # sorting_params["detect_threshold"] = 5.5
        
        # ##################
        
        # # This parameter determines the minimum allowable time interval between detected spikes in the same spatial region (see detect_channel_radius). 
        # # This is useful for avoiding double-detection of individual spikes. The default value is 0.5 msec for scheme 1. The first phase of scheme 2 uses scheme 1,
        # # and in that setting the default value is 1.5 msec in order to avoid including overlapping spikes in the intitial clustering pass.
         
        # sorting_params["detect_time_radius_msec"] = 1.5  
        
        # ##################
        
        # # This parameter is a companion to detect_time_radius_msec. It determines the spatial radius (in units of the channel locations) 
        # # within which the detection time radius applies. In general, a spike is detected at timepoint t on channel m if it is the largest amplitude 
        # # event within a time window size detect_time_radius_msec around t and within a spatial neighborhood around channel m defined by detect_channel_radius. 
        # # The default is None which effectively gives an infinite spatial radius.
        # # This is just like the phase1_detect_channel_radius parameter, but it is used for phase 2.
        
        # sorting_params["scheme2_detect_channel_radius"] = 50
        
        # #################
        
        # #sorting_params["scheme2_max_num_snippets_per_training_batch"] = 200
        
        # # The detect_channel_radius parameter for phase 1
        
        # sorting_params["scheme2_phase1_detect_channel_radius"] = 50  ## I don't want more channels condiered during detection. These are not close enought to 
        #                                                              ## a neuron to be detected simultanoeusly by more channels.
        
        # #################
        
        
        # # This parameter affects dimension reduction prior to clustering. If the dataset has M channels, then M * npca_per_channel PCA features are
        # # computed for each spike snippet. The default value is 3, which is appropriate for most recordings.
        
        # sorting_params["npca_per_channel"] = 3
        
        # #################
        
        # # MountainSort utilizes a subdivision method for clustering. After extracting spike snippets from the preprocessed traces, the data undergo 
        # # dimension reduction through PCA before clustering. Initially, npca_per_subdivision PCA features are computed, followed by Isosplit clustering.
        # # If more than one cluster is detected, the dataset is split into two subdivisions, and feature extraction is performed again within each subdivision
        # # separately, followed by Isosplit clustering to subdivide the clusters further. This process is repeated recursively until each leaf subdivision
        # # returns a single cluster. At each stage, the same number of PCA components (npca_per_subdivision) is used. Recomputing features on each subdivision
        # # offers an advantage, as it allows the refined components to capture features that can better differentiate between clusters within the same overall
        # # feature space region.
        
        # sorting_params["npca_per_subdivision"] = 7 # 7
        
        # #################
        
        
        
        # # The duration of the training data (in seconds). See also the training_recording_sampling_mode parameter.
        
        # # sorting_params["scheme2_training_duration_sec"] = int(dataDurationSec/4) #1/4 of the recording is used as training data
        
        # #################
        
        
        # # How to sample the training data. If 'initial', then the first training_duration_sec of the recording will be used. If 'uniform', then the training 
        # # data will be sampled uniformly in 10-second chunks from the recording.
        
        # sorting_params["scheme2_training_recording_sampling_mode"] = 'uniform'
        
        # #################
        
        
        # # These parameters determine the time duration (units of samples) of the extracted spike snippets. The total duration of the snippet is 
        # # snippet_T1 + snippet_T2 samples, with snippet_T1 samples before the peak and snippet_T2 samples after the peak.
        
        # sorting_params["snippet_T1"] = int(0.4*(samplingRate/1000)) #0.8
        # sorting_params["snippet_T2"] = int(1.5*(samplingRate/1000))
        
        # #################
        
        
        # # If specified, this parameter determines the spatial radius (in units of the channel locations) of a mask that is applied to the extracted spike
        # # snippets. For spikes centered at channel m the mask includes all channels within snippet_mask_radius of channel m, and the signal outside of the
        # # mask region is set to zero. This is useful for removing signal from time-overlapping, spatially-separated events during the clustering phase.
        
        # sorting_params["snippet_mask_radius"] = 60
        
        # #################
        
        # sorting_params["whiten"] = False
        # sorting_params["filter"] = False
        # sorting_params["delete_temporary_recording"] = True
        
        
    #     ####### SCHEME 3 ALGORITHM: refer to https://github.com/flatironinstitute/mountainsort5/blob/main/docs/scheme2.md
        
    # '''
    #     See Scheme1SortingParameters for more details on the parameters below.
    
    #     - phase1_detect_channel_radius: detect_channel_radius in phase 1
    #     - detect_channel_radius: detect_channel_radius in phase 2
    #     - phase1_detect_threshold: detect_threshold in phase 1
    #     - phase1_detect_time_radius_msec: detect_time_radius_msec in phase 1
    #     - detect_time_radius_msec: detect_time_radius_msec in phase 2
    #     - phase1_npca_per_channel: npca_per_channel in phase 1
    #     - phase1_npca_per_subdivision: npca_per_subdivision in phase 1
    #     - detect_sign
    #     - detect_threshold: detect_threshold in phase 2
    #     - snippet_T1
    #     - snippet_T2
    #     - snippet_mask_radius
    #     - max_num_snippets_per_training_batch: the maximum number of snippets to use for training the classifier in each batch
    #     - classifier_npca: the number of principal components to use for each neighborhood classifier
    #     - training_duration_sec: the duration of the training data (in seconds)
    #     - training_recording_sampling_mode: how to sample the training data. If 'initial', then the first training_duration_sec of the recording will be used. If 'uniform', then the training data will be sampled uniformly in 10-second chunks from the recording.
    #     - classification_chunk_sec: the duration of each chunk of data to use for classification (in seconds)
        
        
    #     SCHEME 2 ALGORITHM:
        
    #     Phase 1. First, a subset of the recording is extracted for training, according to the training_duration_sec and training_recording_sampling_mode 
    #     parameters. Then, scheme 1 detection and clustering is performed on this training subset.
        
    #     Phase 2. In this phase, a series of classifiers, one for each channel, are trained using the snippets and labels obtained from phase 1.
    #     The procedure used here is difficult to describe, so for now I will refer the reader to the documented source code in sorting_scheme2.py.
        
    #     Phase 3. Once the classifiers have been fit, new events are detected throughout the entire recording using the detect_threshold and 
    #     detect_channel_radius parameters (possibly different from phase1_detect_threshold and phase1_detect_channel_radius). In order to detect and 
    #     sort more events that overlap in time, it is desirable to choose detect_channel_radius to be smaller than phase1_detect_channel_radius, so that events 
    #     not included in phase 1 can be included in this final phase. The classifiers trained in phase 2 are then used to label each event. Again, for details
    #     we refer to the reader to the documented source code in sorting_scheme2.py.
        
    
    # '''
    
        detect_threshold = 6
    #------------- Run core sorter -------------
    
        sorting_parameters2 = ms5.Scheme2SortingParameters(
            phase1_detect_channel_radius=150,
            detect_channel_radius=60,
            training_duration_sec=50,
            phase1_detect_threshold=detect_threshold,
            phase1_detect_time_radius_msec=1.5,
            detect_time_radius_msec=0.5,
            phase1_npca_per_channel=3,
            phase1_npca_per_subdivision=10,
            phase1_pairwise_merge_step=False,
            detect_sign=0,
            detect_threshold=detect_threshold,
            snippet_T1=int(1*(samplingRate/1000)),
            snippet_T2=int(2*(samplingRate/1000)),
            snippet_mask_radius=80,
            max_num_snippets_per_training_batch=200,
            classifier_npca=None,
            training_recording_sampling_mode='uniform',
            classification_chunk_sec=None
        )
    
    # Create a temporary directory manually
        
        # tmpdir = tempfile.mkdtemp()
        # print(f"Temporary directory created at: {tmpdir}")
    
    
        # try:
        # cache the recording to a temporary directory for efficient reading
        # recording_cached = create_cached_recording(recording_WT, folder=tmpdir)
    
        # use scheme 2
        sorting_MS5 = ms5.sorting_scheme2(
            recording=recording_WT,
            sorting_parameters=sorting_parameters2,
        )
    
    
        #------------- Run spike interf. sorter -------------
        
        
        # #The first two channels are of calibration...
        # bad_ch = [0,1]
        # recording_WT_ = recording_WT.remove_channels(bad_ch)
        
        # sorting_MS5 = ss.run_sorter(
        #     sorter_name='mountainsort5', recording=recording_WT_,**sorting_params,remove_existing_folder = True 
        # )
        
        
        # # sorting_MS5 = ss.run_sorter(
        # #     sorter_name='mountainsort5', recording=recording_WT_, output_folder="Mountainsort5_output",**sorting_params,remove_existing_folder = True,
        # # )
        
        
        print(sorting_MS5)
        
        #%
        #------------- POST PROCESSING -------------
        # Create analyszer
        
        
        # Generate the probe variable
        
     
        
        # Plot the probe
        # plot_probe(probe_2d)
                 
        
        # --- analyzer ---
        analyzer_string = f"analyzer{wellID}{it}"
        
        # # Check if the folder exists and manually delete it
        # if os.path.exists(analyzer_string):
        #     shutil.rmtree(analyzer_string)
       
        
        analyzer = si.create_sorting_analyzer(sorting=sorting_MS5 , recording=recording_WT.set_probe(probe_2d), format='binary_folder', folder=analyzer_string, overwrite=True)
        
        # core extensions
        analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels","unit_locations", "spike_locations", "template_metrics"])
        analyzer.compute('principal_components', n_components = 5, mode="by_channel_local")
        # more extensions
        analyzer.compute(['spike_amplitudes', 'correlograms', 'template_similarity', 'quality_metrics'])
          
        #%                      
        metrics = sqm.compute_quality_metrics(analyzer, metric_names=["snr", "isi_violation", "nearest_neighbor"])
        
        
       
        # unit_ids = sorting_MS5.unit_ids[:Number_electrodes]
        
        
        
        # #%
        # # PLOT SOME RESULTS
        # sw.plot_all_amplitudes_distributions(analyzer, figsize=(10, 10))
        # plt.show()
        
        # # oN THE PROBE
        
        # sw.plot_unit_locations(analyzer, figsize=(4, 8))
        
        # plt.show()
        
        
        # sw.plot_rasters(analyzer)
        # plt.show()
        
        
       
        # ### LUSSAC MERGING
        
        
        # ## Eplanation of the Lussac method refer to: https://doi.org/10.1101/2022.02.08.479192;
        
        # ## API in SpikeInterface: https://spikeinterface.readthedocs.io/en/stable/api.html#spikeinterface.curation.get_potential_auto_merge
        
        # ## Parameter example in Lussac documentation: https://github.com/BarbourLab/lussac/blob/main/src/lussac/params_examples/params_cerebellar_cortex.json
        
        
        # Lussac_merged = scur.get_potential_auto_merge(sorting_analyzer = analyzer, preset = 'similarity_correlograms',
        #                                                    resolve_graph = False, min_spikes = int(200), min_snr =  float(5), 
        #                                                    max_distance_um = float(40.0), corr_diff_thresh = float(0.16), 
        #                                                    template_diff_thresh = float(0.5), contamination_thresh = float(0.35), 
        #                                                    presence_distance_thresh = float(100), p_value = float(0.2), cc_thresh = float(0.1), 
        #                                                    censored_period_ms = float(0.2), refractory_period_ms = float(1.0), 
        #                                                    sigma_smooth_ms = float(0.6), adaptative_window_thresh = float(0.5), 
        #                                                    censor_correlograms_ms = float(0.2), firing_contamination_balance = float(2.5))
                                                           
        
        
        
       
        # ## After Lussac 
        
        
        # # COntrol the data format
        # ext = analyzer.get_extension("spike_amplitudes")
        # ext_data = ext.get_data()
        
        
        # # Merge the proposed units
        # if Lussac_merged:
        
        #     curation_dict = {'removed_units':[],'merge_unit_groups':Lussac_merged,'manual_labels':[]}
            
            
        #     sorting_MS5_SNR_merged = scur.apply_curation(analyzer, **curation_dict, censor_ms=None, new_id_strategy='take_first', merging_mode='hard', sparsity_overlap=0.75, verbose=False)
        # else:
        #     sorting_MS5_SNR_merged =   analyzer  
            
        sorting_MS5_SNR_merged = analyzer
        
        ##### In this section are filtered out the units whose MFR is out [1 250] range
        
        ## Compute firing rates
        firing_rates = sqm.compute_firing_rates(sorting_MS5_SNR_merged)
        FR = firing_rates.values()
        
        ## Get numpy array of the units' firing rate
        FR = np.array(list(FR))
        
        ## Get the idx of the units to keep
        keep_mask = np.where((FR >= 1) & (FR <= 250))
        # print(keep_mask)
        
        ## Get the kept units' id.
        keys = list(firing_rates.keys())
        keep_unit_ids = [keys[i] for i in keep_mask[0]]
        # print(keep_unit_ids)
        
        ## Select units
        sorting_MS5_curated = sorting_MS5_SNR_merged.select_units(keep_unit_ids)
        
        
        
        # print(sorting_MS5_curated)
        
        
        sw.plot_rasters(sorting_MS5_curated)
        # plt.show()
        # sw.plot_unit_locations(sorting_MS5_curated, figsize=(4, 8))
        
        # plt.show()
        
    
        
        
        Sorting_Object = sorting_MS5_curated.sorting
        
       
        # --- Part 1: Construct the Raster Matrix ---
        
        # We'll use a list to collect the new rows, which is more efficient
        # than repeatedly calling np.vstack on a NumPy array.
        raster_rows = []
        number = 1
        
        # Assuming `keep_unit_ids` is an iterable of unit IDs and
        # `Sorting_Object` is a valid object with a `get_unit_spike_train` method.
        for unit_id in keep_unit_ids:
              spiketrain = Sorting_Object.get_unit_spike_train(unit_id)
            
            # Create a column of the current unit number
              unit_column = np.full(len(spiketrain), number)
              
              # Combine the unit number column and the spiketrain into a new array
              new_row = np.column_stack((unit_column, spiketrain))
              
              # Append the new row to our list
              raster_rows.append(new_row)
                
              number += 1
        Raster_array = np.vstack(raster_rows)
        
        # --- Part 2: Define Unit Locations ---
        
        # Assuming `spost.compute_unit_locations` is a function that returns 
        # 2D or 3D locations.
        # We slice the result to get the first two columns (X and Y coordinates).
        # This code block looks good as it is.
        Unit_Location2D = np.array(spost.compute_unit_locations(sorting_MS5_curated)[:,0:2])
        
        
        # Update unit number
        num_unit_temp = np.max(Raster_array[:,0])
        Raster_array[:,0] = Raster_array[:,0]+summed_units
        
    
        # Update summed units
        summed_units = summed_units+num_unit_temp
        
        
        Raster_complete.append(Raster_array)
        Position_complete.append(Unit_Location2D)
        
        
        # Explicitly delete all objects that might hold file handles
      
        del analyzer
        del sorting_MS5
        del Sorting_Object
            
        # finally:
           
        #     # Manually clean up the temporary directory
        #     shutil.rmtree(tmpdir)
        #     print("Temporary directory and its contents have been removed.")
                
        
        
        it = it+1
        
    
        ## SAVE THE RASTER AND UNIT POSITION VALUES
        
        
        
        
        
        
        ## SUMMARY MATRICIES: account for estimated false positives and false negatives for each unit.
        ## Ref: doi: 10.1523/JNEUROSCI.0971-11.2011.
        
        
        
        ## POPULATION LEVEL QUALITY METRIC: FDR
        ## Ref: doi: https://doi.org/10.1101/2023.12.21.572882.
        
        
        
        ##### EXPORT
        
        # ## Investigate the nature of the w_rs variable.
        # w_rs = sw.plot_rasters(curated_sorting) ## Hopefully loads into the variable the raster informations
        
        
        # the export process
        # export_report(sorting_analyzer=
        # sorting_MS5_curated, output_folder=basedir_out)
    
    
    
        
     
     
     
     
     
     
    
    
    # sorting_MS5_curated.save_as(folder="Sorting_analyzer") # we shold get the spatial position of units
    
    
    os.chdir(basedir_out)
    # Save the first array to 'data1.csv'
    np.savetxt(f'Raster_array_{detect_threshold}_{wellID}.csv', np.vstack(Raster_complete), delimiter=',')
    
    # Save the second array to 'data2.csv'
    np.savetxt(f'Unit_Location{detect_threshold}_{wellID}.csv', np.vstack(Position_complete), delimiter=',')


