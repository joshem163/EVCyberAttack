import os
import networkx as nx
import numpy as np
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

import numpy as np
from collections import Counter
from models import Average, make_timeseries, select_threshold, Topo_Fe_TimeSeries_MP, classify_sublist
#Load dataset

import pickle
with open('data/WeekdayScenarios.pkl', 'rb') as g:
    P0Data = pickle.load(g)
N_Scenario=300

#Creat label list
Class=[]
Label=[]
for i in range(N_Scenario):
    Class.append(P0Data[i]['Stations Attacked'])
Label = [classify_sublist(sublist) for sublist in Class]
# Construct networkX graph
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

# Load extracted topological feature

with open('MP_Fe/EV_New_weekday0_150_betti0.data', 'rb') as f:
    mp_data = pickle.load(f)
with open('MP_Fe/EV_New_weekday150_300_betti0.data', 'rb') as f:
    mp_data300 = pickle.load(f)
data_mp = mp_data + mp_data300

# Normalize the Input feature

N_Senario=300
from sklearn.preprocessing import MinMaxScaler
X0=np.array(data_mp)
y=Label

X=[]
for i in range(N_Senario):

    scaler = MinMaxScaler()

# Fit scaler to data and transform data
    X.append(scaler.fit_transform(X0[i]))

# Convert data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

num_samples = N_Senario
num_timesteps = 288
#num_features = len(F_voltage)*len(F_Flow)
num_features = len(mp_data[0][0])
num_classes = len(np.unique(y))




# Define the Transformer model
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_heads, n_layers):
        super(TransformerClassifier, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 288, hidden_dim))  # Assuming max sequence length is 100
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=n_heads, num_encoder_layers=n_layers, num_decoder_layers=n_layers)
        self.fc = nn.Linear(hidden_dim * num_timesteps, output_dim)  # Flatten the output of the transformer

    def forward(self, src):
        src_emb = self.embedding(src) + self.positional_encoding[:, :src.size(1), :]
        src_emb = src_emb.permute(1, 0, 2)  # (seq_len, batch, feature)
        transformer_output = self.transformer.encoder(src_emb)
        transformer_output = transformer_output.permute(1, 0, 2).contiguous().view(src.size(0), -1)  # Flatten
        predictions = self.fc(transformer_output)
        return predictions
def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

input_dim = num_features
hidden_dim = 64
output_dim = num_classes
n_heads = 2
n_layers = 2

# Initialize model, loss function, and optimizer
model = TransformerClassifier(input_dim, hidden_dim, output_dim, n_heads, n_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


loss_per_fold = []
acc_per_fold = []
run = 1

for run in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # Split data


    # Create DataLoader
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # Train the model
    reset_weights(model)
    model.train()
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            output = model(X_batch)
            loss = criterion(output, y_batch)
            test_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    loss_per_fold.append(test_loss / len(test_loader))
    acc_per_fold.append(accuracy)

    print(f'Score for run {run}: Test Loss of {test_loss / len(test_loader)}; Test Accuracy of {accuracy}%')
    run += 1

# Print results
print('Average loss: %.4f' % (np.mean(loss_per_fold)))
print('Average accuracy: %.4f %%' % (np.mean(acc_per_fold)))