import torch
import numpy as np
import networkx as nx
from attrdict import AttrDict
from torch_geometric.data import Data
import torch_geometric as pyg
from torch_geometric.utils import from_networkx, to_networkx, remove_self_loops, to_undirected, grid
from neuraldrawer.network.preprocessing import preprocess_dataset
from neuraldrawer.datasets.transforms import convert_for_DeepGD
import json 
from neuraldrawer.network.train import get_model, go_to_coarser_graph
import neuraldrawer.datasets.datasets as datasets
import os
import deepgd.deepgd as dgd
import time
from visualisation.suitesparse_loader import get_suitesparse, create_pyg_graph
from tqdm import tqdm
import matplotlib.pyplot as plt
import s_gd2

import pygsp as gsp
from torch_geometric.utils.convert import to_scipy_sparse_matrix, from_scipy_sparse_matrix
from neuraldrawer.datasets.transforms import convert_for_DeepGD, convert_for_stress, pmds_layout, filter_connected 
from graph_coarsening.coarsening_utils import *
import graph_coarsening.graph_utils
from torch_geometric.data import Data
from torch_geometric.utils.random import erdos_renyi_graph
import pandas as pd
import pickle
from neuraldrawer.network.losses import ScaledStress
loss_fun = ScaledStress




import rpy2
from rpy2.robjects.packages import importr


def load_model_and_config():
    model_name = 'checkpoints/core_delaunay_no_coarsening.pt'
    config_name = 'configs/core_delaunay_no_coarsening.pt'
    with open(config_name, 'r') as f:
        config = AttrDict(json.load(f))
    eval_model = get_model(config)
    eval_model.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
    eval_model.eval()
    return eval_model, config

def load_deepgd(PATH="deepgd_checkpoints/delauany.pt", device='cpu'):
    model_deepgd = dgd.DeepGD().to(device)
    model_deepgd.load_state_dict(torch.load(PATH, map_location=torch.device(device)))
    return model_deepgd


def get_PivotMDS(G):
    igraph = importr('igraph', robject_translations={'.env': '__env'})
    ggraph = importr('ggraph')
    graphlayouts = importr('graphlayouts')
    edges_from, edges_to = G.edge_index[0].tolist(), G.edge_index[1].tolist()
    edges = [None]*(len(edges_from)+len(edges_to))
    edges[::2] = edges_from
    edges[1::2] = edges_to
    edges = [vertex + 1 for vertex in edges]
    edge_vector = rpy2.robjects.IntVector(edges)
    graph = igraph.make_undirected_graph(edges=edge_vector)
    graph = igraph.simplify(graph)
    pos_pivotmds = graphlayouts.layout_with_pmds(graph, pivots=min(50, G.x.shape[0]))
    pos_x = torch.unsqueeze(torch.tensor(list(pos_pivotmds.rx(True, 1))), dim=-1)
    pos_y = torch.unsqueeze(torch.tensor(list(pos_pivotmds.rx(True, 2))), dim=-1)
    torch.cat((pos_x, pos_y), dim=1)

def coarsen_dataset(config, dataset):
    dataset_gsp = [gsp.graphs.Graph(to_scipy_sparse_matrix(G.edge_index)) for G in dataset]

    method = config.coarsen_algo 
    r    = config.coarsen_r 
    k    = config.coarsen_k  
        
    if True:
        coarsened_pyg = []
        coarsening_matrices = []
        for i in range(len(dataset)):
            pyg_graphs = [dataset[i]]
            matrices = []
            while pyg_graphs[-1].edge_index[0].max() > config.coarsen_min_size:
                C, Gc, Call, Gall = coarsen(gsp.graphs.Graph(to_scipy_sparse_matrix(pyg_graphs[-1].edge_index)), K=k, r=r, method=method, max_levels=1)
                pyg_graphs.append(Data(edge_index=from_scipy_sparse_matrix(Gall[1].W)[0]))
                matrices.append(Call[0])
            pyg_graphs[-1].x = torch.zeros(pyg_graphs[-1].edge_index[0].max()+1)
            coarsened_pyg.append(pyg_graphs)

            for i in range(len(matrices)):
                matrices[i][matrices[i] > 0] = 1.0
                matrices[i] = matrices[i].tocoo()
                matrices[i] = torch.sparse.LongTensor(torch.LongTensor([matrices[i].row.tolist(), matrices[i].col.tolist()]),
                                torch.FloatTensor(matrices[i].data))
            coarsening_matrices.append(matrices) 
        preprocessed_dataset = preprocess_dataset([coarsened[-1] for coarsened in coarsened_pyg], config)
        for i in range(len(preprocessed_dataset)):
            preprocessed_dataset[i].index = i
            preprocessed_dataset[i].coarsening_level = 0
        
        return preprocessed_dataset, coarsened_pyg, coarsening_matrices

def go_to_coarser_graph(graph, last_embeddings, device, batch, coarsened_graphs, coarsening_matrices, noise):
    new_level = graph.coarsening_level+1
    embeddings_finer = torch.transpose(torch.sparse.mm(torch.transpose(last_embeddings, 0, 1), coarsening_matrices[graph.index][-new_level].to(device)), 0, 1)
    graph.edge_index = coarsened_graphs[graph.index][-new_level-1].edge_index.to(device)
    graph.x = embeddings_finer
    if batch:
        graph.batch = torch.zeros(embeddings_finer.shape[0], device=device, dtype=torch.int64)
    # add noise to embeddings
    mean = 0
    std = noise
    graph.x = graph.x + torch.tensor(np.random.normal(mean, std, graph.x.size()), dtype=torch.float, device=device)
    graph.coarsening_level = new_level
    return graph


def run_core(config, graph, model, layer_num):
    model.eval()
    model = model.to(device='cuda')
    batch = dataset[0]
    batch = batch.to(device='cuda')
    pred, states = model(batch, layer_num, return_layers=True)
    if config.coarsen:
        for i in range(1, len(coarsening_matrices[batch.index])+1):
            batch = go_to_coarser_graph(batch, states[-1],'cuda',  True, coarsened_graphs, coarsening_matrices, noise=config.coarsen_noise)

            pred, states = model(batch, layer_num, encode=False, return_layers=True)

    return pred


if __name__ == "__main__":
    node_sizes = list(range(100, 1000, 10)) + list(range(1000, 10000, 100)) + list(range(10000, 100000, 1000))
    nd_model, config = load_model_and_config()
    nd_model.eval()
    nd_model = nd_model.to('cuda:0')
    dg_model = load_deepgd().to(device='cuda')
    dg_model.eval()
    
    times = []

    dgd_preprocess_failed = False

    with torch.no_grad():
        for n in tqdm(node_sizes):
            for seed in range(5):
                G = datasets.random_delaunay_graph(n)

                #Inference
                start_time = time.perf_counter() 
                dataset_nd = preprocess_dataset([G], config)
                dataset_nd[0].to('cuda:0')
                pred = nd_model(dataset_nd[0], 20)
                end_time = time.perf_counter() 
                times.append(['CoRe-GD-no-coarsening', n, seed, end_time - start_time])

                start_time = time.per_counter()
                dataset_nd = coarsen_dataset(config, dataset_nd)
                dataset_nd[0].to('cuda:0')
                pred = run_core(config, dataset_nd[0], nd_model, 10)
                end_time = time.perf_counter()
                times.append(['CoRe-GD-coarsening', n, seed, end_time - start_time])

                start_time = time.perf_counter() 
                G_dgd = convert_for_DeepGD(G).to(device=0)
                dg_model(G_dgd)
                end_time = time.perf_counter() 
                times.append(['DeepGD', n, seed, end_time - start_time])

                I = G.edge_index[0].tolist()
                J = G.edge_index[1].tolist()
                start_time = time.perf_counter()
                s_gd2.layout(I, J)
                end_time = time.perf_counter() 
                times.append(['sgd2', n, seed, end_time - start_time])

                start_time = time.perf_counter() 
                get_PivotMDS(G)
                end_time = time.perf_counter() 
                times.append(['PivotMDS', n, seed, end_time - start_time])

                nx_graph = to_networkx(G, to_undirected=True)
                start_time = time.perf_counter()
                nx.drawing.nx_agraph.graphviz_layout(nx_graph, prog='neato', args='-Gdimen=2')
                end_time = time.perf_counter() 
                times.append(['neato', n, seed, end_time - start_time])

                nx_graph = to_networkx(G, to_undirected=True)
                start_time = time.perf_counter()
                nx.drawing.nx_agraph.graphviz_layout(nx_graph, prog='sfdp', args='-Gdimen=2')
                end_time = time.perf_counter() 
                times.append(['sfdp', n, seed, end_time - start_time])

                del G
            df = pd.DataFrame(times, columns =['model', 'n', 'seed', 'time'])
            df.to_csv('runtimes.csv')

    print("#####")
    print(times)
    df = pd.DataFrame(times, columns =['model', 'n', 'seed', 'time'])
    df.to_csv('runtimes.csv')
