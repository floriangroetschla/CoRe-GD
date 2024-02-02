# CoRe-GD

## Setup

Create the conda environment
```
CONDA_OVERRIDE_CUDA=11.7 conda env create -f environment.yml
conda activate CoRe-GD

# Install pyg
pip install torch==2.0.0
pip install torch_geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html

# Other dependencies
pip install test-tube wandb s_gd2 cupy-cuda11x ssgetpy

# Create directory to store models
mkdir models

# Install the neuraldrawer package that provides CoRe-GD
pip install -e .
```

# Install the graph coarsening implementation used in the paper
```
it clone https://github.com/SergeiShumilin/graph-coarsening-up-to-date.git
cd graph-coarsening-up-to-date
pip install -e .
cd ..
```

If you want to use DeepGD, you need a valid R installation (including the python interface `rpy2`) with the `igraph` and `graphlayouts` packages. These are needed to compute PivotMDS initializations. You can do so as follows:

```
conda install r-essentials r-base
pip install rpy2
R   # This will let you enter the R shell
> install.packages("igraph")
> install.packages("graphlayouts")
> quit()
```

You also have to clone DeepGD-Demo into the root of the repo
```
git clone https://github.com/yolandalalala/DeepGD-Demo.git deepgd
```


## Training
Training for CoRe-GD can be done by providing a config file. Configs for all dataset runs and ablation studies are provided in `configs/`:
```
python run_experiment.py --config configs/config_rome.json
```

Note: Logging is done through wandb


Assuming that you installed the necessary dependencies to compute PivotMDS initializations:
```
mkdir deepgd_models
python run_deepgd.py --dataset rome
```

## Evaluation
During training, checkpoints are stored in the `models/` directory. Pretrained checkpoints are available for all datasets in `checkpoints/`.
In addition to the checkpoint, a config file is also needed. For the provided checkpoints you can use the default configs in `configs/` for the datasets.
Lastly, we specify the dataset to test on:
```
python eval_CoRe-GD.py checkpoints/core_rome.pt configs/config_rome.json rome 
```

For DeepGD, we provide checkpoints in `deepgd_checkpoints/`. They can be run on a dataset with:
```
python eval_dgd.py deepgd_checkpoints/rome.pt rome
```

To run the non-learned baselines on all datasets (5 runs with different random seeds each), do as follows:
```
python eval_baselines.py
```
This will store a csv containing all run information in `baseline_runs.csv`

The results should align with the following values:
| Model       | Rome              | ZINC             | MNIST             | CIFAR10          | PATTERN            | CLUSTER            |
| :---        |    :----:         |    :----:        |    :----:         |    :----:        |    :----:          |    :----:          |
| PivotMDS    | 388.77 &pm;  1.02 | 29.85 &pm;  0.00 | 173.32 &pm;  0.12 | 384.64 &pm; 0.20 | 3813.30 &pm;  1.89 | 3538.25 &pm;  0.73 |
| neato       | 244.22  &pm; 0.55 |  5.76 &pm;  0.05 | 129.87 &pm;  0.19 | 263.44 &pm; 0.20 | 3188.34 &pm;  0.59 | 2920.30 &pm;  0.75 |
| sfdp        | 296.05 &pm;  1.16 | 20.02 &pm;  0.27 | 172.63 &pm;  0.11 | 377.97 &pm; 0.11 | 3219.23 &pm;  1.31 | 2952.95 &pm; 1.81  |
| (sgd)$^2$   | 233.49 &pm;  0.22 | 5.14 &pm;  0.01  | 129.19 &pm;  0.00 | 262.52 &pm; 0.00 | 3181.51 &pm;  0.05 | 2920.36 &pm;  0.78 |
| DeepGD      | 235.22 &pm; 0.71  | 6.19 &pm; 0.07   | 129.23 &pm; 0.03  | 262.91 &pm; 0.13 | 3080.70 &pm; 0.24  | 2838.13 &pm; 0.08  |
| CoRe-GD     | 233.17 &pm;  0.13 | 5.11 &pm;  0.02  | 129.10 &pm;  0.02 | 262.68 &pm; 0.08 | 3067.02 &pm;  0.79 | 2827.81 &pm;  0.36 |
| CoRe-GD-mix | 234.60 &pm; 0.10  | 5.21 &pm; 0.02   | 129.24 &pm; 0.01  | 262.90 &pm; 0.02 | 3066.31 &pm; 0.20  | 2828.13 &pm; 0.12  |


## Scaling runs
The runtimes for the scaling experiment from the paper can be reproduced with 
```
python runtime_comparison.py
```

