#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wandb', type=bool, default=None, help='enables wandb logging and disables local logfiles')
    parser.add_argument("--wandb_project", type=str, default=None, help='specifies wandb project to log to')
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help='specifies wandb username to team name where the project resides')
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="set run name to differentiate runs, if you don't set this wandb will auto generate one")
    # primary run settings
    parser.add_argument('--model', type=str, default=None, help='set model, resnet or squeezenet')
    parser.add_argument('--pretrained', type=int, default=None, help='use pretrained model if 1, random init of 0')
    parser.add_argument('--ncm', type=int, default=None, help='ncm initialization if 1, otherwise no ncm init')
    parser.add_argument('--algorithm', type=str, default=None, help='which algorithm to run (ft or lp')
    parser.add_argument('--fl_algorithm', type=str, default=None, help='which fl algorithm to run (fedavg, fedadam, fedavgm')
    parser.add_argument('--epochs', type=int, default=None, help="number of rounds of training")
    parser.add_argument('--num_clients', type=int, default=None, help="number of clients: K")
    parser.add_argument('--clients_per_round', type=int, default=None, help='number of clients selected for each round')
    parser.add_argument('--local_ep', type=int, default=None, help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=None, help="local batch size: B")
    parser.add_argument('--client_lr', type=float, default=None, help='learning rate for client models')
    parser.add_argument('--server_lr', type=float, default=None, help='learning rate for server optimizer')
    parser.add_argument('--momentum', type=float, default=None, help='momentum for fedavgm')
    parser.add_argument('--dataset', type=str, default=None, help="name of dataset.")
    parser.add_argument('--optimizer', type=str, default=None, help="type of optimizer (adam or sgd)")
    parser.add_argument('--alpha', type=float, default=None, help="alpha of dirichlet, value between 0 and infinity\
                        more homogeneous when higher, more heterogeneous when lower")
    parser.add_argument('--num_client_samples', type=int, default=None, help='set a specific number of samples to '
                                                                             'distribute to each client')
    args = parser.parse_args()
    return args
