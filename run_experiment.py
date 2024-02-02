from email.policy import default
from test_tube import HyperOptArgumentParser
from test_tube.hpc import SlurmCluster
import json
import hashlib

from neuraldrawer.network.train import train_and_eval

if __name__ == "__main__":
    parser = HyperOptArgumentParser(strategy='grid_search')
    parser.add_argument('--device', type=int, default=0)
    parser.opt_list('--hidden_dimension', type=int, default=64, tunable=True, options=[64])
    parser.opt_list('--dropout', type=float, default=0.0, tunable=True, options=[0.0])
    parser.opt_list('--run_number', type=int, default=1, tunable=True, options=[1])
    parser.opt_list('--iter_mean', type=float, default=5.0, tunable=True, options=[5.0])
    parser.opt_list('--iter_var', type=float, default=1, tunable=True, options=[1])
    parser.opt_list('--lr', type=float, default=0.0002, tunable=True, options=[0.0002])
    parser.add_argument('--epochs', type=int, default=200)
    parser.opt_list('--batch_size', type=int, default=32, tunable=True, options=[32])
    parser.opt_list('--conv', type=str, default='gru', tunable=True, options=['gru'])
    parser.opt_list('--skip_previous', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--skip_input', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--use_beacons', type=bool, default=True, tunable=True, options=[True])
    parser.opt_list('--laplace_eigvec', type=int, default=8, tunable=True, options=[8])
    parser.opt_list('--random_in_channels', type=int, default=1, tunable=True, options=[1])
    parser.opt_list('--hidden_state_factor', type=float, default=4, tunable=True, options=[4])
    parser.opt_list('--mlp_depth', type=float, default=2, tunable=True, options=[2])
    parser.opt_list('--weight_decay', type=float, default=0.0, tunable=True, options=[0.0])
    parser.opt_list('--dataset', type=str, default='rome', tunable=True, options=['rome'])
    parser.opt_list('--aggregation', type=str, default='add', tunable=True, options=['add'])
    parser.opt_list('--use_entropy_loss', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--normalization', type=str, default='LayerNorm', tunable=True, options=['LayerNorm'])
    parser.opt_list('--rewiring', type=str, default='knn', tunable=True, options=['knn'])
    parser.opt_list('--knn_k', type=int, default=8, tunable=True, options=[8])
    parser.opt_list('--replay_train_replacement_prob', type=float, default=0.5, tunable=True, options=[0.5])
    parser.opt_list('--replay_buffer_replacement_prob', type=float, default=1.0, tunable=True, options=[1.0])
    parser.opt_list('--alt_freq', type=int, default=2, tunable=True, options=[2]) 
    parser.opt_list('--num_beacons', type=int, default=2, tunable=True, options=[2])
    parser.opt_list('--encoding_size_per_beacon', type=int, default=8, tunable=True, options=[8])
    parser.opt_list('--out_dim', type=int, default=2, tunable=True, options=[2])
    parser.opt_list('--replay_buffer_size', type=int, default=4096, tunable=True, options=[4096])
    parser.opt_list('--coarsen', type=bool, default=False, tunable=True, options=[False])
    parser.opt_list('--coarsen_prob', type=float, default=0.5, tunable=True, options=[0.8])
    parser.opt_list('--num_replay_batches', type=int, default=8, tunable=True, options=[8])
    parser.opt_list('--coarsen_noise', type=float, default=0.01, tunable=True, options=[0.1])
    parser.opt_list('--coarsen_k', type=int, default=5, tunable=True, options=[5])
    parser.opt_list('--coarsen_r', type=float, default=0.8, tunable=True, options=[0.8])
    parser.opt_list('--coarsen_algo', type=str, default='heavy_edge', tunable=True, options=['heavy_edge'])
    parser.opt_list('--coarsen_min_size', type=int, default=50, tunable=True, options=[100])
    parser.add_argument('--use_replay_buffer', type=bool, default=True)
    parser.add_argument('--randomize_between_epochs', type=bool, default=True)
    parser.add_argument('--store_models', type=bool, default=True)
    parser.add_argument('--slurm', action='store_true', default=True)
    parser.add_argument('--use_cupy', type=bool, default=False)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--verbose', type=bool, default=True)
    parser.add_argument('--wandb_project_name', type=str, default="CoRe-GD")
    args = parser.parse_args()

    if args.model_name is None:
        hash_object = hashlib.sha256(json.dumps(args.__getstate__()).encode())
        model_hash = hash_object.hexdigest()
        args.model_name = model_hash

    if args.config is not None:
        with open(args.config, 'r') as f:
            conf = json.load(f)
            for key in conf.keys():
                setattr(args, key, conf[key])
    
    if args.slurm:
        cluster = SlurmCluster(
            hyperparam_optimizer=args,
            log_path='slurm_log/',
            python_cmd='python'
        )
        cluster.job_time = '48:00:00'

        args.filename = ""
        cluster.memory_mb_per_node = '60G'
        job_name = 'NeuralDrawer'
        cluster.per_experiment_nb_cpus = 4
        cluster.per_experiment_nb_gpus = 1
        cluster.optimize_parallel_cluster_gpu(train_and_eval, nb_trials=None, job_name=job_name, job_display_name='NeuralDrawer')
    else:
        train_and_eval(args)
