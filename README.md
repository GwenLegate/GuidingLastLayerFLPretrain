
# Guiding The Last Layer in Federated Learning with Pre-Trained Models

This repository contains the source code for our paper "Guiding The Last Layer in Federated Learning with Pre-Trained Models"
where we investigate transfer learning in a federated setting. Our work builds off of 
[Where to Begin? Exploring the Impact of Pre-Training and Initialization in Federated Learning (Nguyen et. al. 2022)](https://arxiv.org/abs/2206.15387)
and our implementation modifies the [FL Sim](https://github.com/facebookresearch/FLSim) source code. 

### How to Set up a Run:
FL Sim sets up run configurations using `config.py`, additionally we have implemented the option to configure run settings
using the command line. To see `config.py` <=> command line equivalences, see method `set_cfg_from_cl()` in `utils.py`.
 If you do not supply a command line argument, configuration will defer to the value set in `config.py`.

#### &nbsp;&nbsp;Sample Run Command for FT:
`python federated_main.py --wandb=False --epochs=100 --num_clients=10 --clients_per_round=10 --dataset=cifar --local_ep=3
--pretrained=1 --ncm=0 --algorithm=ft --fl_algorithm=fedavg --optimizer=sgd --alpha=0.1 --client_lr=0.001`
#### &nbsp;&nbsp;Sample Run Command for FedNCM+FT:
`python federated_main.py --wandb=False --epochs=100 --num_clients=10 --clients_per_round=10 --dataset=cifar --local_ep=3
--pretrained=1 --ncm=1 --algorithm=ft --fl_algorithm=fedavg --optimizer=sgd --alpha=0.1 --client_lr=0.001`

training results for sample run command for ft (blue) and sample run command for FedNCM+FT (yellow)
![alt text](https://github.com/GwenLegate/GuidingLastLayerFLPretrain/blob/main/images/fl_base.png?raw=true)
#### &nbsp;&nbsp;wandb:
This code base works with wandb logging, to enable it, set the appropriate command line options, or the configs in the 
wandb section of `config.py`.

#### &nbsp;&nbsp;Some Command line Options:
|Option                |Args                                      |Comments                              |
|----------------------|------------------------------------------|--------------------------------------|
| `--model`            |resnet, squeezenet                        |                                      |
|`--pretrained`        |1 (True), 0 (False)                       |                                      |
|`--ncm`               |1 (True), 0 (False)                       |                                      |
|`--mu`                |hyperparameter used with fedprox option   |                                      |
|`--algorithm`         |ft, lp, fedprox                           |                                      |
|`--fl_algorithm`      |fedavg, fedadam, fedavgm                  |                                      |
|`--momentum`          |float                                     |server momentum                       |
|`--optimizer`         |sgd, adam                                 |                                      |
|`--alpha`             |float                                     | min=0.01                             |
|`--dataset`           |flowers, cifar, cars, cub, eurosat        |Only a fraction of Eurosat is selected|
|`--num_client_samples`|int                                       |                                      |
|`--client_lr`         |float                                     |                                      |
|`--server_lr`         |float                                     |always set to 1 for fedavg            |
|`--local_ep`          |int                                       |number of client epochs               |
|`--local_bs`          |int                                       |batch size for local training         |
|`--epochs`            |int                                       |global rounds                         |
|`--num_clients`       |int                                       |                                      |
|`--clients_per_round` |int                                       |Note: FL Sim automatically scales global rounds to client fraction (see **)|

** Round scaling: If you have 50 epochs, 10 clients and 5 clients per round you will end up running a total of (10/5)*50 
global rounds in total. If you want to remove this behavior, the code will need to be modified appropriately.