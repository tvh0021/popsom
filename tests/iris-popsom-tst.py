import pandas as pd
import numpy as np
import argparse
import h5py as h5

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# import os
# os.chdir('/Users/tha/git_repos')

import sys
sys.path.append('/Users/tha/git_repos')

# print("Current paths")
# print(sys.path, flush=True)

import popsom.popsom as popsom


# CLI arguments
parser = argparse.ArgumentParser(description='popsom test on iris dataset')
parser.add_argument('--xdim', type=int, dest='xdim', default=10, help='Map x size')
parser.add_argument('--ydim', type=int, dest='ydim', default=10, help='Map y size')
parser.add_argument('--alpha', type=float, dest='alpha', default=0.5, help='Learning parameter')
parser.add_argument('--train', type=int, dest='train', default=10000, help='Number of training steps')

args = parser.parse_args()
xdim = args.xdim
ydim = args.ydim
alpha = args.alpha
train = args.train

# Load and preprocess data
iris = datasets.load_iris()

iris_data = iris.data
feature_list = iris.feature_names

scale = False
if scale == True:
        scaler = StandardScaler()
        scaler.fit(iris_data)
        x = scaler.transform(iris_data)
else:
        x = iris_data


attr=pd.DataFrame(x)
attr.columns=feature_list
# print(attr.head(), flush=True)

# Train SOM
print(f'constructing full SOM for xdim={xdim}, ydim={ydim}, alpha={alpha}, adaptive train...', flush=True)
m=popsom.map(xdim, ydim, alpha, train)
 
labels = np.array(list(range(len(x))))
m.fit(attr,labels)

term = m.final_epoch
print("Final epoch", term, flush=True)

# print changes in neuron weights
neuron_weights = m.weight_history
np.save(f'evolution_iris_{xdim}{ydim}_{alpha}_{term}.npy', neuron_weights, allow_pickle=True)
# np.set_printoptions(threshold=np.inf)
# print("Neurons weight sum history", flush=True)
# print(m.weight_history, flush=True)

neurons = m.all_neurons()
np.save(f'neurons_iris_{xdim}{ydim}_{alpha}_{term}.npy', neurons, allow_pickle=True)

#Data matrix with neuron positions:
print("Calculating projection")
data_matrix=m.projection()
data_Xneuron=data_matrix[:,0]
data_Yneuron=data_matrix[:,1]
print("data matrix: ", flush=True)
print(data_matrix[:10,:], flush=True)
print("Printing Xneuron info", flush=True)
print("Shape of Xneuron: ", data_Xneuron.shape, flush=True)
print("Printing Yneuron info", flush=True)
print("Shape of Yneuron: ", data_Yneuron.shape, flush=True)

#Neuron matrix with centroids:
umat = m.compute_umat(smoothing=2)
centrs = m.compute_combined_clusters(umat, False, 0.15) #0.15
centr_x = centrs['centroid_x']
centr_y = centrs['centroid_y']

#create list of centroid _locations
neuron_x, neuron_y = np.shape(centr_x)

centr_locs = []
for i in range(neuron_x):
        for j in range(neuron_y):
                cx = centr_x[i,j]
                cy = centr_y[i,j]

                centr_locs.append((cx,cy))

unique_ids = list(set(centr_locs))
# print(unique_ids)
n_clusters = len(unique_ids)
print("Number of clusters", flush=True)
print(n_clusters)

mapping = {}
for I, key in enumerate(unique_ids):
        # print(key, I)
        mapping[key] = I

clusters = np.zeros((neuron_x,neuron_y))
for i in range(neuron_x):
        for j in range(neuron_y):
                key = (centr_x[i,j], centr_y[i,j])
                I = mapping[key]

                clusters[i,j] = I

print("clusters", flush=True)
print(clusters, flush=True)

print("Assigning clusters", flush=True)
        
cluster_id = np.zeros((len(x)))

for i in range(len(x)):
    cluster_id[i] =  clusters[int(data_Xneuron[i]), int(data_Yneuron[i])]

f5 = h5.File(f'clusters_iris_{xdim}{ydim}_{alpha}_{term}.h5', 'a')
dsetx = f5.create_dataset("cluster_id",  data=cluster_id)
dsety = f5.create_dataset("true_id",  data=iris.target)
f5.close()
print("Done writing the cluster ID file")



