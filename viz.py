import json
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import random
from matplotlib.widgets import Button, TextBox
from utils import load_jsons_from_ids, load_challenges_single_dict

# Reformulate viz.py so it can be called with just a list of task ids, like show_id_list.
# Move things internal to the visualizer class as much as possible, e.g. make comments_filepath
# a default argument and it should load the file (or create it) on its own. It can use functions
# from utils.py. When you're finished the file should consist of only imports and the visualizer
# class. The visualizer interface should look the exact same - instead of listing the number out
# of 400, it should list the number out of the list it was called with, e.g. should pop up on
# 1/50 if it is called with 50 ids. It should still show the id for each plot.

class Visualizer:
    def __init__(self, probs_idx_list, title='', comments_file='comments.json'):
        self.probs_idx_list = probs_idx_list
        self.title = title
        self.comments_file = comments_file

        if os.path.exists(comments_file):
            with open(comments_file, 'r') as f:
                self.comments = json.load(f)
        else:
            self.comments = {task: '' for task in load_challenges_single_dict()} # Problem hash : blank comment

        self.color_map = {
                            0: '#000000', 1: '#0074D9', 2: '#FF4136', 3: '#2ECC40', 4: '#FFDC00',
                            5: '#AAAAAA', 6: '#F012BE', 7: '#FF851B', 8: '#7FDBFF', 9: '#870C25'
                        }

        self.tasks = load_jsons_from_ids(probs_idx_list)

        self.current_set = 0
        self.fig = plt.figure(figsize=(16, 9))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        
        # Add random button
        self.random_button_ax = plt.axes([0.45, 0.02, 0.1, 0.04])
        self.random_button = Button(self.random_button_ax, 'Random')
        self.random_button.on_clicked(self.random_problem)
        
        # Add text input box
        self.text_box_ax = plt.axes([0.1, 0.93, 0.8, 0.03])
        self.text_box = TextBox(self.text_box_ax, 'Notes: ', initial='')
        
        self.plot_current_set()
        plt.show()

    def plot_current_set(self):
        self.save_current_note()

        self.fig.clear()
        task = self.tasks[self.current_set]
        n_cases = len(task['train']) + len(task['test'])
        
        # Determine grid layout based on number of cases
        if n_cases <= 3:
            n_rows, n_cols = 2, 3
        elif n_cases <= 5:
            n_rows, n_cols = 2, 5
        elif n_cases <= 10:
            n_rows, n_cols = 4, 5
        else:
            n_rows, n_cols = 6, 5
        
        # Create GridSpec with appropriate spacing
        gs = self.fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.2)
        
        # Plot train cases
        for i, train_case in enumerate(task['train']):
            row, col = divmod(i, n_cols)
            ax_input = self.fig.add_subplot(gs[row * 2, col])
            self.plot_grid(ax_input, train_case['input'], f'Train In {i+1}')
            ax_output = self.fig.add_subplot(gs[row * 2 + 1, col])
            self.plot_grid(ax_output, train_case['output'], f'Train Out {i+1}')
        
        # Plot test cases
        for i, test_case in enumerate(task['test']):
            idx = i + len(task['train'])
            row, col = divmod(idx, n_cols)
            ax_input = self.fig.add_subplot(gs[row * 2, col])
            self.plot_grid(ax_input, test_case['input'], f'Test In {i+1}')
            #ax_output = self.fig.add_subplot(gs[row * 2 + 1, col])
            #self.plot_grid(ax_output, test_case['output'], f'Test Out {i+1}')
        
        self.fig.suptitle(f'{self.title} Prob {self.current_set + 1} of {len(self.probs_idx_list)}: {self.probs_idx_list[self.current_set]}', fontsize=16, y=0.98)
        
        # Recreate text input box with current note
        self.current_file = self.probs_idx_list[self.current_set]
        self.text_box_ax = plt.axes([0.1, 0.93, 0.8, 0.03])
        self.text_box = TextBox(self.text_box_ax, 'Notes: ', initial=self.comments[self.current_file.replace('.json','')])
        
        # Recreate random button
        self.random_button_ax = plt.axes([0.45, 0.02, 0.1, 0.04])
        self.random_button = Button(self.random_button_ax, 'Random')
        self.random_button.on_clicked(self.random_problem)
        
        plt.tight_layout(rect=[0, 0.07, 1, 0.91])
        self.fig.canvas.draw()

    def plot_grid(self, ax, grid, title):
        rgb_grid = self.grid_to_rgb(np.array(grid))
        ax.pcolormesh(np.arange(len(grid[0])+1), np.arange(len(grid)+1), rgb_grid[::-1], edgecolors='w', linewidth=0.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.set_xlim(0, len(grid[0]))
        ax.set_ylim(0, len(grid))
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)

    def grid_to_rgb(self, grid):
        rgb_grid = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                rgb_grid[i, j] = [int(self.color_map[grid[i, j]][k:k+2], 16) for k in (1, 3, 5)]
        return rgb_grid

    def on_key_press(self, event):
        if event.key == 'right':
            self.current_set = (self.current_set + 1) % len(self.probs_idx_list)
            self.plot_current_set()
        elif event.key == 'left':
            self.current_set = (self.current_set - 1) % len(self.probs_idx_list)
            self.plot_current_set()

    def random_problem(self, event):
        self.current_set = random.randint(0, len(self.probs_idx_list) - 1)
        self.plot_current_set()

    def save_current_note(self):
        if hasattr(self, 'current_file'):
            self.comments[self.current_file.replace('.json','')] = self.text_box.text
        self.save_comments_to_file()

    def save_comments_to_file(self):
        with open(self.comments_file, 'w') as f:
            json.dump(self.comments, f, indent=2)

    def on_close(self, event):
        self.save_current_note()