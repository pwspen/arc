# plotting from https://www.kaggle.com/code/wrinkledtime/quick-look-at-train-data

import matplotlib.pyplot as plt
from   matplotlib import colors
import numpy as np
from collections import Counter
from matplotlib.colors import ListedColormap
import json

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

def load_all_tasks():
    base_path = 'arc/data/'
    training_challenges =  load_json(base_path +'arc-agi_training_challenges.json')
    training_solutions =   load_json(base_path +'arc-agi_training_solutions.json')
    evaluation_challenges =load_json(base_path +'arc-agi_evaluation_challenges.json')
    evaluation_solutions = load_json(base_path +'arc-agi_evaluation_solutions.json')
    return training_challenges, training_solutions, evaluation_challenges, evaluation_solutions

def load_challenges_single_dict():
    d1, _, d2, _ = load_all_tasks()
    return {**d1, **d2}

def load_jsons_from_ids(ids):
    tasks = load_challenges_single_dict()
    samples = []
    for id in ids:
        samples.append(tasks[id])
    return samples

def plot_one(task, ax, i, train_or_test, input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap='viridis', norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
   
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)

def plot_sample(sample, fig, axs, id):
    n_samples = len(sample['train'])
    
    for i in range(n_samples):  
        plot_one(sample, axs[0,i], i, 'train', 'input')
        plot_one(sample, axs[1,i], i, 'train', 'output')        
    
    plot_one(sample, axs[2,0], 0, 'test', 'input')
    
    fig.suptitle(f"ID: {id}")
    fig.canvas.draw_idle()

def show_id_list(ids, title=''):
    samples = load_jsons_from_ids(ids)
    current_index = 0
    
    fig, axs = plt.subplots(3, max(len(sample['train']) for sample in samples), figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    def plot_current_sample():
        for ax in axs.flat:
            ax.clear()
        plot_sample(samples[current_index], fig, axs, ids[current_index])
    
    def on_key(event):
        nonlocal current_index
        if event.key == 'right' and current_index < len(samples) - 1:
            current_index += 1
        elif event.key == 'left' and current_index > 0:
            current_index -= 1
        else:
            return
        plot_current_sample()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plot_current_sample()
    plt.tight_layout()
    plt.show()

def create_sub_barplot(ax, data, title, x_label, y_label):
    # Count occurrences of each unique value
    count_dict = Counter(data)
    
    # Sort the dictionary by keys
    sorted_counts = dict(sorted(count_dict.items()))
    
    # Create bar plot
    bars = ax.bar(sorted_counts.keys(), sorted_counts.values())
    
    # Set title and labels
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    
    # Set x-axis ticks
    ax.set_xticks(list(sorted_counts.keys()))
    
    # Add gridlines
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Adjust y-axis limits
    y_max = max(sorted_counts.values())
    ax.set_ylim(0, y_max * 1.1)  # Set upper limit to 110% of max value
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height,
                f'{height}', ha='center', va='bottom')

def create_sub_heatmap(ax, data, title, x_label, y_label):
    # Count occurrences of each unique tuple
    count_dict = Counter(data)
    
    # Find the maximum x and y values
    max_x = max(x for x, y in count_dict.keys()) + 1
    max_y = max(y for x, y in count_dict.keys()) + 1
    
    # Create a 2D array to hold the counts
    heatmap_data = np.zeros((max_y, max_x))
    
    # Fill the array with counts
    for (x, y), count in count_dict.items():
        heatmap_data[y, x] = count
    
    viridis = plt.cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))
    newcolors[0, :] = np.array([0, 0, 0, 1])  # Make the first color black
    custom_cmap = ListedColormap(newcolors)

    # Create heatmap
    im = ax.imshow(heatmap_data, cmap=custom_cmap, interpolation='nearest', aspect='auto', origin='lower')
    
    # Set title and labels
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    
    # Set ticks and labels
    x_ticks = np.arange(0, max_x, max(1, max_x // 6))
    y_ticks = np.arange(0, max_y, max(1, max_y // 6))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xticklabels(x_ticks)
    ax.set_yticklabels(y_ticks)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.grid(which='major', color='white', linestyle='-', linewidth=0.5)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Count", rotation=-90, va="bottom")
    
    # Loop over data dimensions and create text annotations
    for i in range(max_y):
        for j in range(max_x):
            if heatmap_data[i, j] > 0:
                color = "white" if heatmap_data[i, j] > np.max(heatmap_data) / 2 else "black"
                
    
    # Adjust layout
    ax.figure.tight_layout()

# Example usage with two subplots
def create_barplot(list1, title1, x_label1, y_label1):
    # Create figure with two subplots
    fig, (ax1) = plt.subplots(1, 1, figsize=(16, 6))
    
    # Create first subplot
    create_sub_barplot(ax1, list1, title1, x_label1, y_label1)
    
    # Adjust layout and display plot
    plt.tight_layout()
    plt.show()

def aprint(arr):
    print(np.array_str(np.array(arr), precision=2, suppress_small=True))