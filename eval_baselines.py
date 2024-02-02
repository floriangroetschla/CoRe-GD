import torch
import json
from attrdict import AttrDict
import torch_geometric as pyg
import networkx as nx
from torch_geometric.utils import to_networkx
import json 
from neuraldrawer.network.losses import ScaledStress
import neuraldrawer
import s_gd2
import networkx as nx
import statistics
import neuraldrawer.datasets.datasets as datasets
from tqdm import tqdm
from torch_geometric.data import Data, Batch

import rpy2
from rpy2.robjects.packages import importr
import networkx as nx
import pandas as pd


import warnings
warnings.filterwarnings('ignore')

loss_fun = ScaledStress

def get_loss(loader, algo, seed=0):
    if algo == 'pmds':
        loss, loss_normalized = get_pivotmds_losses(loader, pivots=50)
    elif algo == 'neato':
        loss, loss_normalized = get_graphviz_losses(loader, engine='neato', seed=seed)
    elif algo == 'sfdp':
        loss, loss_normalized = get_graphviz_losses(loader, engine='sfdp', seed=seed)
    elif algo == 'sgd2':
        loss, loss_normalized = get_s_gd2_losses(loader, seed=seed)
    return statistics.mean(loss), statistics.mean(loss_normalized)

def get_s_gd2_losses(loader, seed=0):
    stress = loss_fun()
    s_gd2_losses = []
    s_gd2_losses_normalized = []
    for batch in tqdm(loader, 'sgd2'):
        I = batch.edge_index[0].tolist()
        J = batch.edge_index[1].tolist()
        X = s_gd2.layout(I, J, random_seed=seed)
        stress_s_gd2 = stress(torch.tensor(X), batch).item()
        stress_s_gd2_normalized = stress_s_gd2 / (batch.x.shape[0]*batch.x.shape[0])
        s_gd2_losses.append(stress_s_gd2)
        s_gd2_losses_normalized.append(stress_s_gd2_normalized)
    return s_gd2_losses, s_gd2_losses_normalized

def get_graphviz_losses(loader, engine='neato', dim=2, seed=0):
    stress = loss_fun()
    graphviz_losses = []
    graphviz_losses_normalized = []
    for batch in tqdm(loader, engine):
        nx_graph = to_networkx(batch, to_undirected=True)
        pos_graphviz = nx.drawing.nx_agraph.graphviz_layout(nx_graph, prog=engine, args='-Gstart=' + str(seed) + ' -Gdimen=' + str(dim))
        stress_graphviz = stress(torch.tensor(list(pos_graphviz.values())), batch).item()
        stress_graphviz_normalized = stress_graphviz / (batch.x.shape[0]*batch.x.shape[0])
        graphviz_losses.append(stress_graphviz)
        graphviz_losses_normalized.append(stress_graphviz_normalized)
    return graphviz_losses, graphviz_losses_normalized

def get_pivotmds_losses(loader, pivots=10):
    stress = loss_fun()
    pivotmds_losses = []
    pivotmds_losses_normalized = []
    igraph = importr('igraph', robject_translations={'.env': '__env'})
    graphlayouts = importr('graphlayouts')
    for batch in tqdm(loader, 'pmds'):
        edges_from, edges_to = batch.edge_index[0].tolist(), batch.edge_index[1].tolist()
        edges = [None]*(len(edges_from)+len(edges_to))
        edges[::2] = edges_from
        edges[1::2] = edges_to
        edges = [vertex + 1 for vertex in edges]
        edge_vector = rpy2.robjects.IntVector(edges)
        graph = igraph.make_undirected_graph(edges=edge_vector)
        graph = igraph.simplify(graph)
        pos_pivotmds = graphlayouts.layout_with_pmds(graph, pivots=min(pivots, batch.x.shape[0]))
        pos_x = torch.unsqueeze(torch.tensor(list(pos_pivotmds.rx(True, 1))), dim=-1)
        pos_y = torch.unsqueeze(torch.tensor(list(pos_pivotmds.rx(True, 2))), dim=-1)
        stress_pivotmds = stress(torch.cat((pos_x, pos_y), dim=1), batch).item()
        stress_pivotmds_normalized = stress_pivotmds / (batch.x.shape[0]*batch.x.shape[0])
        pivotmds_losses.append(stress_pivotmds)
        pivotmds_losses_normalized.append(stress_pivotmds_normalized)
    return pivotmds_losses, pivotmds_losses_normalized


if __name__ == "__main__":
    datasets = ['rome', 'zinc', 'cifar10', 'pattern', 'cluster', 'mnist']
    algos = ['pmds', 'neato', 'sfdp', 'sgd2']
    seeds = range(5)
    results = []

    for dataset in datasets:
        _, _, dataset_nd = neuraldrawer.datasets.datasets.get_dataset(dataset)
        loader = pyg.loader.DataLoader(dataset_nd, batch_size=1, shuffle=False)
        for algo in algos:
            for seed in seeds:
                loss, loss_normalized = get_loss(loader, algo, seed)
                results.append([dataset, algo, seed, loss, loss_normalized])

    df = pd.DataFrame(results, columns=['dataset', 'algo', 'seed', 'loss', 'loss_normalized'])
    df.to_csv('baseline_runs.csv')
