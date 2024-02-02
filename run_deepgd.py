from email.policy import default
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster

import torch
import torch_geometric as pyg
from tqdm.auto import *

import deepgd.deepgd as dgd
import tqdm
import numpy as np
from neuraldrawer.datasets.datasets import get_dataset

def main(config, cluster=None):
    device = "cpu"
    for backend, device_name in {
        torch.backends.mps: "mps",
        torch.cuda: "cuda",
    }.items():
        if backend.is_available():
            device = device_name

    train, val, test = get_dataset(config.dataset, for_DeepGD=True)

    model = dgd.DeepGD().to(device)
    criteria = {
        dgd.Stress(): 1,
        dgd.EdgeVar(): 0,
        dgd.Occlusion(): 0,
        dgd.IncidentAngle(): 0,
        dgd.TSNEScore(): 0,
    }
    optim = torch.optim.AdamW(model.parameters())

    batch_size = config.batch_size

    train_loader = pyg.loader.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = pyg.loader.DataLoader(val, batch_size=1, shuffle=False)

    best_val_loss = 10000

    model_name = config.dataset + '_' + str(config.run_number)

    for epoch in range(1000):
        model.train()
        losses = []
        for batch in tqdm.tqdm(train_loader):
            batch = batch.to(device)
            model.zero_grad()
            loss = 0
            for c, w in criteria.items():
                loss += w * c(model(batch), batch)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        print(f'[Epoch {epoch}] Train Loss: {np.mean(losses)}')
        with torch.no_grad():
            model.eval()
            losses = []
            for batch in tqdm.tqdm(val_loader, disable=True):
                batch = batch.to(device)
                loss = 0
                for c, w in criteria.items():
                    loss += w * c(model(batch), batch)
                losses.append(loss.item())
            print(f'[Epoch {epoch}] Val Loss: {np.mean(losses)}')
        if np.mean(losses) < best_val_loss:
            best_val_loss = np.mean(losses)
            torch.save(model.state_dict(), 'deepgd_models/' + model_name + '_best_val.pt')

    torch.save(model.state_dict(), 'deepgd_models/' + model_name + '_last.pt') 


if __name__ == "__main__":
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.opt_list('--dataset', type=str, default='rome', tunable=True, options=['rome'])
    parser.opt_list('--run_number', type=int, default=1, tunable=True, options=[1,2,3,4,5])
    parser.opt_list('--batch_size', type=int, default=32, tunable=True, option=[32])
    parser.add_argument('--slurm', action='store_true', default=False)
    args = parser.parse_args()
    
    if args.slurm:
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='slurm_log/',
            python_cmd='python'
        )
        cluster.job_time = '48:00:00'

        args.filename = ""
        cluster.memory_mb_per_node = '60G'
        job_name = 'DeepGD'
        cluster.per_experiment_nb_cpus = 16
        cluster.per_experiment_nb_gpus = 1
        cluster.optimize_parallel_cluster_gpu(main, nb_trials=None, job_name=job_name, job_display_name='DeepGD')
    else:
        main(args)



