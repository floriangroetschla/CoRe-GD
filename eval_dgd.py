import torch
import statistics
import neuraldrawer.datasets.datasets as datasets
import torch_geometric as pyg

from tqdm import tqdm
from torch_geometric.data import Data, Batch
from neuraldrawer.network.losses import ScaledStress


import deepgd.deepgd as dgd
import sys

loss_fun = ScaledStress

def get_DeepGD_losses(loader, eval_model):
    eval_model = eval_model.to(device)
    with torch.no_grad():
        stress = loss_fun()
        deepgd_losses = []
        deepgd_losses_normalized = []
        for batch in tqdm(loader, 'DeepGD'):
            batch = batch.to(device)
            pred = eval_model(batch)
            stress_deepgd = stress(pred, batch).item()
            deepgd_losses.append(stress_deepgd)
            deepgd_losses_normalized.append(stress_deepgd/(batch.x.shape[0]*batch.x.shape[0]))
    return deepgd_losses, deepgd_losses_normalized


if __name__ == "__main__":
    device = "cpu"
    for backend, device_name in {
        torch.backends.mps: "mps",
        torch.cuda: "cuda",
    }.items():
        if backend.is_available():
            device = device_name

    model_path = sys.argv[1]
    dataset_name = sys.argv[2]

    _, _, dataset_deepgd = datasets.get_dataset(dataset_name, for_DeepGD=True)

    model_deepgd = dgd.DeepGD().to(device)
    model_deepgd.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model_deepgd.eval()

    loader_deepgd = pyg.loader.DataLoader(dataset_deepgd, batch_size=1, shuffle=False)

    losses, losses_normalized = get_DeepGD_losses(loader_deepgd, model_deepgd)
    print(f'Mean loss: {statistics.mean(losses)}, Mean normalized loss: {statistics.mean(losses_normalized)}')

