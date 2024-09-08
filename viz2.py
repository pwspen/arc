import json
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import os
import random
from matplotlib.widgets import Button, TextBox

# Folder containing JSON files
json_folder = 'evaluation'  # Replace with your actual folder path

# Load all JSON files from the folder
data_sets = []
json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
for json_file in json_files:
    with open(os.path.join(json_folder, json_file), 'r') as file:
        data_sets.append(json.load(file))

# Load or create comments file
comments_file = 'comments.json'
if os.path.exists(comments_file):
    with open(comments_file, 'r') as f:
        comments = json.load(f)
else:
    comments = {json_file.replace('.json', ''): "" for json_file in json_files} # Problem hash : blank comment

for i in data_sets:
    print(i)