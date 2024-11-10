import os
import networkx as nx
import multiprocessing as mp

import pandas as pd
import numpy as np
from collections import Counter
from models import Average,Topo_Fe_TimeSeries_MP,make_timeseries,select_threshold
import multiprocessing
from joblib import Parallel, delayed

# import dataset
import pickle
with open('data/WeekdayScenarios.pkl', 'rb') as g:
    P0Data = pickle.load(g)
N_Scenario=len(P0Data)

# Construct NetworkX graph
node=P0Data[0]['Bus Voltages'].keys()
Edge=P0Data[0]['Branch flows'].keys()

# Initialize an empty graph
G = nx.Graph()  # or nx.DiGraph() for a directed graph
# Add nodes (convert dict_keys to a list if needed)
G.add_nodes_from(list(node))

# Add edges
G.add_edges_from(Edge)
N=G.nodes()
E=G.edges()
N=list(G.nodes)
E=list(G.edges)
time_step=288

# Select Threshold value for node voltage and brach flow

voltage_threshold = np.array([0.34, 0.35, 0.94, 0.98, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05])

BFlow_threshold = np.array([8.8e-01, 1.75e+00, 7.24e+00, 2.453e+01, 5.928e+01,
                            1.3625e+02, 3.9623e+02, 5.0359e+02, 1.29343e+03,
                            1.33055e+03, 1.60132e+03, 1.66341e+03, 1.73374e+03])

def extract_topological_features(graph_id):
    time_series=make_timeseries(graph_id,time_step)
    betti=Topo_Fe_TimeSeries_MP(time_series[0],time_series[1], voltage_threshold,BFlow_threshold)
    return betti

def process_graph(graph_id):
    # Function to extract topological features from a single graph
    return extract_topological_features(graph_id)

# topological feature for the graph mention in range()


result=Parallel(n_jobs=multiprocessing.cpu_count()-2)(
        delayed(process_graph)(graph_id)
            for graph_id in range(150,300)

    )
print("150 to 300")
with open('EV_paralal_weekday150300_betti0.data', 'wb') as f:
    pickle.dump(result, f)
