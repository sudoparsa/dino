import numpy as np
import os
from PIL import Image
from tqdm import tqdm

def data(X, y, path):
    for i, img in enumerate(tqdm(X)):
        im = Image.fromarray(img)
        im.save(os.path.join(path, str(y[i]), str(i)+'.jpg'))
folder = 'bcmnist8'
X_train = np.load(open(os.path.join(folder, "X_train.npy"), "rb"))
y_train = np.load(open(os.path.join(folder, "y_train.npy"), "rb"))
env_train = np.load(open(os.path.join(folder, "env_train.npy"), "rb"))
X_test = np.load(open(os.path.join(folder, "X_test.npy"), "rb"))
y_test = np.load(open(os.path.join(folder, "y_test.npy"), "rb"))
env_test = np.load(open(os.path.join(folder, "env_test.npy"), "rb"))
X_val = np.load(open(os.path.join(folder, "X_val.npy"), "rb"))
y_val = np.load(open(os.path.join(folder, "y_val.npy"), "rb"))
env_val = np.load(open(os.path.join(folder, "env_val.npy"), "rb"))

folder = 'bcmnist8_dino'
os.mkdir(folder)
os.mkdir(os.path.join(folder, 'train'))
os.mkdir(os.path.join(folder, 'val'))
os.mkdir(os.path.join(folder, 'test'))
for i in range(10):
    os.mkdir(os.path.join(folder, 'train', str(i)))
for i in range(10):
    os.mkdir(os.path.join(folder, 'val', str(i)))
for i in range(10):
    os.mkdir(os.path.join(folder, 'test', str(i)))

data(X_train, y_train, os.path.join(folder, 'train'))
data(X_val, y_val, os.path.join(folder, 'val'))
data(X_val, y_val, os.path.join(folder, 'test'))