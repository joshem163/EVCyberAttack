import os
import networkx as nx
import numpy as np
import pandas as pd
import math
import pyflagser

import pickle
with open('data/WeekdayScenarios.pkl', 'rb') as g:
    P0Data = pickle.load(g)
N_Senario=len(P0Data)

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

def Average(lst):
    return sum(lst) / len(lst)
def make_timeseries(scenario, total_time_step):
    time_series_voltage=[]
    time_series_BFlow=[]
    for time_step in range(total_time_step):
        voltage=[]
        B_flow=[]
        for node in N:
            node_voltage=P0Data[scenario]['Bus Voltages'][node][time_step]
            #voltage.append(P0Data[scenario]['Bus Voltages'][node][time_step].tolist())
            voltage.append(Average(node_voltage))
        time_series_voltage.append(voltage)
        for edge in E:
            B_flow.append([P0Data[scenario]['Branch flows'][edge][time_step]])
        time_series_BFlow.append(B_flow)
    return time_series_voltage, time_series_BFlow

def Topo_Fe_TimeSeries_MP(TS_voltage, TS_branchFlow, F_voltage,F_Flow):
    betti_0=[]
    for k in range(len(TS_voltage)):
        fec=[]
        AverageVoltage=[]
        Voltage=TS_voltage[k]
        for y in Voltage:
            AverageVoltage.append(y)
        #AverageVoltage = [i * 100 for i in AverageVoltage]
        BranchFlow=[]
        Branch_Flow=TS_branchFlow[k]
        for j  in range(len(Branch_Flow)):
            BranchFlow.append(Branch_Flow[j][0])

        for p in range(len(F_voltage)):
            Active_node_v=np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
            for q in range(len(F_Flow)):
                #if AverageVoltage[p]> F_voltage[p] and BranchFlow[q]> F_Flow[q]:
                #n_active = np.where(np.array(AverageVoltage) > F_voltage[p])[0].tolist()
                n_active=Active_node_v.copy()
                #print(n_active)
                G = nx.DiGraph()
                G.add_nodes_from(n_active)
                indices = np.where(np.array(BranchFlow) > F_Flow[q])[0].tolist()
                for s in indices:
                    a=int(N.index(E[s][0]))
                    b=int(N.index(E[s][1]))
                    if a in n_active and b in n_active:
                        G.add_edge(a, b)
                    #n_active.append(int(N.index(E[s][0])))
                    #n_active.append(int(N.index(E[s][1])))
                #Active_node=np.unique(n_active)
                #print(G.edges())
                if (len(n_active)==0):
                    fec.append(0)
                else:
                    #b=A[Active_node,:][:,Active_node]
                    Adj = nx.to_numpy_array(G)
                    my_flag=pyflagser.flagser_unweighted(Adj, min_dimension=0, max_dimension=2, directed=False, coeff=2, approximation=None)
                    x = my_flag["betti"]
                    fec.append(x[0])
                n_active.clear()
            Active_node_v.clear()
        betti_0.append(fec)
    return betti_0


import numpy as np
from collections import Counter


def select_threshold(time_series, Min_frq):
    array_2d = np.array(time_series)

    # Flatten the array and count values
    flattened_array = array_2d.flatten()
    value_counts = Counter(flattened_array)
    # print(value_counts.items())

    # Find values with frequency greater than 10
    values_above_10 = [value for value, count in value_counts.items() if count > Min_frq]

    most_frequent = np.sort(values_above_10)
    rounded_array = np.round(most_frequent, 2)
    trashold = np.unique(rounded_array)


    return trashold
def classify_sublist(sublist):
    if sublist == []:
        return 0  # Class 0 for []
    elif sublist == [0]:
        return 1  # Class 1 for [0]
    elif sublist == [1]:
        return 2  # Class 2 for [1]
    elif sublist == [2]:
        return 3  # Class 3 for [2]
    else:
        return -1  # For sublists not matching any class (shouldn't happen)

