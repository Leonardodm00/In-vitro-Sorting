
import numpy as np
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import numpy as np

from kilosort.run_kilosort import load_sorting
        
def plot_probe_and_units(pos,Visible=True):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import math
    """
    Plots the electrode positions of a square probe and overlays the
    2D positions of neural units.

    Args:
        pos (np.ndarray): A NumPy array of shape (N, 2) containing the
                          (x, y) coordinates of the neural units.
    """
    # --- Step 1 & 2: Define Probe Parameters and Generate Electrode Positions ---
    Number_electrodes = 2304
    pitch = 60 # um
    Electrode_size = 25 # um

    k = 0
    n = int(math.sqrt(Number_electrodes))
    electrode_positions = np.zeros((Number_electrodes, 2))
    for j in range(n):
        for i in range(n):
            x = i * pitch
            y = j * pitch
            electrode_positions[k] = x, y
            k = k + 1
            
            
            
    print(np.max(electrode_positions))
    if Visible == True:
        # --- Step 3: Set Up the Plot ---
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal', adjustable='box')
    
        # --- Step 4: Plot the Electrodes ---
        for x_pos, y_pos in electrode_positions:
            # Calculate the bottom-left corner of the square to center it on the position
            rect = patches.Rectangle(
                (x_pos - Electrode_size / 2, y_pos - Electrode_size / 2),
                Electrode_size, Electrode_size,
                linewidth=1,
                edgecolor='gray',
                facecolor='lightgray',
                zorder=1
            )
            ax.add_patch(rect)
    
        # --- Step 5: Plot the Unit Positions ---
        # Use a colormap to plot each unit with a different color
        sc = ax.scatter(pos[:, 0], pos[:, 1], c=np.random.rand(len(pos)), cmap='viridis', s=50, zorder=2)
    
    
        # --- Step 6: Finalize the Plot ---
        ax.set_xlabel('X Position ($\mu$m)')
        ax.set_ylabel('Y Position ($\mu$m)')
        ax.set_title(f'Probe Layout and Unit Positions ({Number_electrodes} Electrodes)')
    
        # Set limits to fit the data
        ax.set_xlim(-Electrode_size / 2, n * pitch + Electrode_size / 2)
        ax.set_ylim(-Electrode_size / 2, n * pitch + Electrode_size / 2)
    
        plt.grid(True)
        plt.show()  
        

    
def curation_extraction(kilo_folders,use_kilo_labels =True,Visible = True):
    from pathlib import Path
    import shutil
    
    import numpy as np
    
    from kilosort.run_kilosort import load_sorting
        
    '''
    The used formalism to construct the raster array is: 1st column = unit idx 
    2nd column is times
    
    
    '''
    
    
    
    unit_positions = []
    unit_spikes = []
    running_unit = 0
    
    
    for pth in kilo_folders:
        
    
    
        section_spks = {}
        section_positions = {}
    
        
        # Path to load existing sorting results from.
        results_dir = Path(pth)
      
        # Load sorting results
        ops, st, clu, similar_templates, is_ref, est_contam_rate, kept_spikes = \
            load_sorting(results_dir)
        
        cluster_labels = np.unique(clu)  # integer label for each cluster
        fs = ops['fs']                   # sampling rate
        
        
        
        
            
            # Option 1: Use existing labels as a starting point.
        #           KS4 assigns "good" where is_ref is True, and "mua" otherwise.
        label_good = is_ref.copy()
        
        
        if use_kilo_labels == False:
            
            n_units = is_ref.size
            
            label_good = np.full(n_units, True, dtype=bool)
            
            
        spk_pos_path =Path(pth) / "spike_positions.npy"     # Combine the path with the filename using the '/' operator

        spk_positions = np.load(spk_pos_path)
        
        # Option 2: Ignore KS4's labels and only use your own criteria. Contamination rate and 
                    # MFR
        
        
        
        contam_good = est_contam_rate < 0.2   # this already has shape (n_clusters,)
        fr_good = np.zeros(cluster_labels.size, dtype=bool)
        for i, c in enumerate(cluster_labels):
            # Get all spikes assigned to this cluster
            spikes = st[clu == c]
            positions = spk_positions[clu == c]
            
            # Calculate the center of mass
            cm = np.mean(positions, axis=0)
            
            section_spks[c] = spikes
            section_positions[c] = cm
            # Compute est. firing rate using your preferred method.
            # Note that this formula will not work well for units that drop in and out.
            if spikes.size > 1:
                fr = spikes.size / (spikes.max() / fs - spikes.min() / fs)
            else:
                fr = 0  # Set firing rate to 0 if there's only one spike or no spikes.
                
            if fr >= 0.1:
                fr_good[i] = True
        
        # Update labels, requiring that all criteria hold for each cluster.
        lab = np.logical_and( contam_good, fr_good)
        label_good = np.logical_and(label_good,lab)
        
        
        good_cluster_indices = np.where(label_good)[0]
        good_labels = cluster_labels[good_cluster_indices]
        
        spk_tostack = []
        pos_tostack = []
        
        # Extract meaningful clusters 
        for i,c in enumerate(good_labels):
            
            spks = section_spks[c]
            
            
            repeated_array = np.full(len(spks), running_unit + i)
            
            
            new_row = np.column_stack((repeated_array, spks))
            spk_tostack.append(new_row)
            pos_tostack.append(section_positions[c])
        
        spk_tostack = np.vstack(spk_tostack)
        pos_tostack = np.vstack(pos_tostack)
        
        if len(unit_positions) == 0:
            unit_positions = pos_tostack
            unit_spikes = spk_tostack
        else:
                
            unit_positions = np.vstack((unit_positions, pos_tostack))
            unit_spikes = np.vstack((unit_spikes, spk_tostack))
        
        running_unit = running_unit + len(good_cluster_indices)
        
    print(f'Total number of units: {running_unit} ')
    
    if Visible == True:
        plot_probe_and_units(unit_positions)
        
    
    return unit_positions,unit_spikes

def plot_raster(spikes: np.ndarray, neuron_labels: list = None, title: str = "Raster Plot"):
    """
    Plots a raster plot from a 2D NumPy array of spike data.

    A raster plot visualizes the spiking activity of a population of neurons
    over time. Each row represents a neuron, and each dot represents a spike.

    Args:
        spikes (np.ndarray): A 2D NumPy array of shape (N, 2) where N is the
                             total number of spikes.
                             The first column (spikes[:, 0]) must contain
                             the neuron index (e.g., 0, 1, 2, ...).
                             The second column (spikes[:, 1]) must contain
                             the spike timing.
        neuron_labels (list, optional): A list of strings to label the y-axis.
                                        If None, the neuron indices will be used.
                                        Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Raster Plot".

    Returns:
        None: The function displays the plot directly.
    """
    # Ensure the input array has the correct shape
    if spikes.shape[1] != 2:
        print("Error: The input array must have a shape of (N, 2).")
        return

    # Extract neuron indices and spike timings
    neuron_indices = spikes[:, 0]
    spike_timings = spikes[:, 1]

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Use a scatter plot for efficiency and clarity
    plt.scatter(spike_timings, neuron_indices, s=5, c='black')

    # Configure the plot
    plt.title(title, fontsize=16)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Neuron Index", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.yticks(np.unique(neuron_indices)) # Ensure y-ticks correspond to neuron indices

    # Optional: set custom neuron labels
    if neuron_labels:
        # Check if the number of labels matches the number of neurons
        num_neurons = len(np.unique(neuron_indices))
        if len(neuron_labels) != num_neurons:
            print(f"Warning: Number of labels ({len(neuron_labels)}) does not match the number of unique neurons ({num_neurons}). Using default labels.")
        else:
            plt.yticks(np.arange(num_neurons), neuron_labels)

    # Invert the y-axis to have neuron 0 at the top
    plt.gca().invert_yaxis()

    # Display the plot
    plt.show()

# --- Example Usage ---


kilo_folders = [r"C:\Users\Admin\Desktop\Leonardo\LOW DATA\400\A1\WithBands\SORTER OUTPUT_lowDens400Well_A1\Kilo_out_0",
                r"C:\Users\Admin\Desktop\Leonardo\LOW DATA\400\A1\WithBands\SORTER OUTPUT_lowDens400Well_A1\Kilo_out_1",
                r"C:\Users\Admin\Desktop\Leonardo\LOW DATA\400\A1\WithBands\SORTER OUTPUT_lowDens400Well_A1\Kilo_out_2",
                r"C:\Users\Admin\Desktop\Leonardo\LOW DATA\400\A1\WithBands\SORTER OUTPUT_lowDens400Well_A1\Kilo_out_3"
]       
        
import os
# %matplotlib
new_directory = r'C:\Users\Admin\Desktop\Leonardo\LOW DATA\400\A1'


pos,spk =  curation_extraction(kilo_folders,use_kilo_labels = True)     
  
%matplotlib
      
plot_raster(spk)
os.chdir(new_directory)
# 3. Save to CSV using numpy.savetxt

# Save the 'pos' array
np.savetxt(
    'pos_data.csv',        # The filename
    pos,                   # The array to save
    delimiter=',',         # Use a comma as the separator (CSV standard)
            # Format floating-point numbers to 5 decimal places
)

# Save the 'spk' array
np.savetxt(
    'spk_data.csv',        # The filename
    spk,                   # The array to save
    delimiter=',',
   
)
