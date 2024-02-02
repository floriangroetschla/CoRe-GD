import torch
import json
from attrdict import AttrDict
import torch_geometric as pyg
import networkx as nx
import json 
from neuraldrawer.network.train import get_model, test
from neuraldrawer.network.losses import ScaledStress
import statistics
from neuraldrawer.network.preprocessing import preprocess_dataset 
import neuraldrawer.datasets.datasets as datasets
from tqdm import tqdm
from torch_geometric.data import Data, Batch

import sys

import warnings
warnings.filterwarnings('ignore')

loss_fun = ScaledStress()
PATH_TO_MODELS = "checkpoints/"

def load_model_and_config(model_path, config_path, device='cpu'):
    with open(config_path, 'r') as f:
        config = AttrDict(json.load(f))
    eval_model = get_model(config)
    eval_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    eval_model.eval()
    return eval_model, config

if __name__ == "__main__":
    model_path = sys.argv[1]
    config_path = sys.argv[2]
    dataset_name = sys.argv[3]
    device = f'cuda' if torch.cuda.is_available() else 'cpu'

    model_nd, config = load_model_and_config(model_path, config_path)
    model_nd = model_nd.to(device)

    data_1, data_2, dataset_nd = datasets.get_dataset(dataset_name)
    dataset_nd = dataset_nd
    for idx in range(len(dataset_nd)):
        dataset_nd[idx] = dataset_nd[idx].to(device)

    if config.coarsen:
        dataset_nd, coarsened, coarsening_matrices = create_coarsened_dataset(config, dataset_nd)
    else:
        dataset_nd = preprocess_dataset(dataset_nd, config)
        coarsened = None
        coarsening_matrices = None
    
    loader_nd = pyg.loader.DataLoader(dataset_nd, batch_size=1, shuffle=False)

    nd_loss = test(model_nd, device, loader_nd, loss_fun, 20, coarsened_graphs=coarsened, coarsening_matrices=coarsening_matrices, coarsen=config.coarsen, noise=config.coarsen_noise)

    print(f'Mean loss: {nd_loss}')
