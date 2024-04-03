import torch
path = f'logs/photoshapes/990000.tar'
# assert file exists
import os
assert os.path.exists(path), f"File {path} does not exist"

ckpt = torch.load(path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
styles = ckpt['styles'].detach().cpu().numpy()

# SHAPE = True
# if SHAPE:
#     styles = styles[:, :styles.shape[1]//2]
# else :
#     #color
#     styles = styles[:, styles.shape[1]//2:]

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Function to read and resize images
def read_and_resize_image(filename, size=(20, 20)):
    img = plt.imread(filename)
    return img

# read instances file
instances_file = f'./data/photoshapes/instances.txt'
with open(instances_file, 'r') as f:
    instances = f.readlines()
    instances = instances[:100]
    instances = [i.strip() for i in instances]

# shape09135_rank02

dir_paths = [f"./data/photoshapes/{i}/" for i in instances]
import os
for i in dir_paths:
    assert os.path.isdir(i), f"Directory {i} does not exist"    

def get_random_file(dir_path):
    files = os.listdir(dir_path + "train/")
    files = [f for f in files if f.endswith('.png')]
    import random
    return random.choice(files)

image_paths = [f"{d}train/{get_random_file(d)}" for d in dir_paths]
for i in image_paths:
    assert os.path.isfile(i), f"File {i} does not exist"

# Load and resize images
images = [read_and_resize_image(img) for img in image_paths]

# apply tsne on styles to get 2d embeddings
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
data_2d = tsne.fit_transform(styles)

# save tsne object
import pickle
with open('tsne.pkl', 'wb') as f:
    pickle.dump(tsne, f)

#load tsne object
# import pickle
# with open('tsne.pkl', 'rb') as f:
#     tsne = pickle.load(f)


# Apply PCA for 2D embeddings
# pca = PCA(n_components=2)
# data_2d = pca.fit_transform(styles)

# save the pca object
# import pickle
# with open('pca.pkl', 'wb') as f:
#     pickle.dump(pca, f)
    
# load the pickle object
# import pickle
# with open('pca.pkl', 'rb') as f:
#     pca = pickle.load(f)
    
# # Plotting
fig, ax = plt.subplots()

for i in range(len(data_2d)):
    # Plot data point
    ax.scatter(data_2d[i, 0], data_2d[i, 1], c='r', marker='x')

    # Display image next to data point
    imagebox = OffsetImage(images[i], zoom=0.06)
    ab = AnnotationBbox(imagebox, (data_2d[i, 0], data_2d[i, 1]), frameon=False, pad=0)
    ax.add_artist(ab)

# Set labels and title

if SHAPE:
    title = 'tsne on shape embeddings'
else :
    title = 'tsne on color embeddings'
title = 'tsne on style embeddings'

plt.title(title)
# plt.savefig("tsne_shape" if SHAPE else "tsne_color", dpi=300)
plt.savefig("tsne" if SHAPE else "tsne_color", dpi=300)
plt.show()
