#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import flsim.configs  # noqa
import torch
from flsim.data.data_sharder import SequentialSharder, RandomSharder
from flsim.channels.base_channel import FLChannelConfig
from flsim.clients.base_client import ClientConfig
from flsim.servers.sync_servers import SyncServerConfig
from flsim.active_user_selectors.simple_user_selector import (
    SequentialActiveUserSelectorConfig, UniformlyRandomActiveUserSelectorConfig
    
)
from flsim.optimizers.local_optimizers import LocalOptimizerSGDConfig, LocalOptimizerAdamConfig
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import fl_config_from_json
from omegaconf import DictConfig, OmegaConf
from config import json_config
import wandb
from torchvision.models import squeezenet1_1, resnet18, ResNet18_Weights
from torchvision.models.squeezenet import SqueezeNet1_1_Weights
from sync_trainer import SyncTrainer, SyncTrainerConfig
import os
from utils import validata_dataset_params, build_data_provider, FLModel, MetricsReporter, inference, set_cfg_from_cl, wandb_setup
import torch.nn.functional as F
from flsim.optimizers.server_optimizers import FedAvgWithLROptimizerConfig, FedAdamOptimizerConfig, FedAvgOptimizerConfig
from utils import DistillBERTClassifier

def main(cfg,
    use_cuda_if_available: bool = True,
) -> None:
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{0}" if cuda_enabled else "cpu")

    if cfg.trainer.pretrained:
        if cfg.wandb.offline:
            if cfg.trainer.model == 'resnet':
                model = resnet18()
                model.load_state_dict(torch.load("./models/pretrained_resnet.pt"))
            elif cfg.trainer.model == "distillbert":
                model = DistillBERTClassifier()
            else:
                model = squeezenet1_1()
                model.load_state_dict(torch.load("./models/pretrained_squeeze.pt"))
        else:
            if cfg.trainer.model == 'resnet':
                model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            elif cfg.trainer.model == "distillbert":
                model = DistillBERTClassifier()
            else:
                model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
    else:
        if cfg.trainer.model == 'resnet':
            model = resnet18()
        else:
            model = squeezenet1_1()

    # 4 algorithms: ft--> fine tune all, lp--> only train last layer, FedNCM--> NCM init only, FedNCM+FT--> NCM init and FT
    if 'lp' in cfg.trainer.algorithm:
        if cfg.trainer.model == "distillbert":
            for name, param in model.distill_bert.named_parameters():
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                param.requires_grad = False

    # replace classifier with randomly initialized classifier of appropriate size
    if cfg.trainer.model == 'resnet':
        model.fc = torch.nn.Linear(512, cfg.dataset.num_classes)
    else:
        model.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(512, cfg.dataset.num_classes)
        )

    model.eval()
    # wandb setup
    if cfg.wandb.activate:
        run_dir = f'/scratch/{os.environ.get("USER", "glegate")}/{cfg.wandb.run_name}'
        run_dir = './'
        if not os.path.isdir(run_dir):
            os.makedirs(run_dir, mode=0o755, exist_ok=True)
        wandb_setup(cfg)
        wandb.watch(model, log_freq=cfg.wandb.log_freq)

    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    fl_data_loader, data_provider = build_data_provider(
        local_batch_size=cfg.trainer.client.local_bs,
        num_clients=cfg.trainer.total_users,
        dataset=cfg.dataset.name,
        num_classes=cfg.dataset.num_classes,
        alpha=cfg.dataset.alpha,
        num_client_samples=cfg.dataset.num_client_samples,
        drop_last=False,
    )
    trainloader = torch.utils.data.DataLoader(
        fl_data_loader.train_dataset, batch_size=cfg.trainer.client.local_bs, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(
        fl_data_loader.test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # NCM initialization
    if cfg.trainer.ncm_init:
        
        if cfg.trainer.model == "distillbert" and cfg.dataset.name == "rotten_tomatoes":
            class_sums = torch.zeros((2,768)).to(device)

            labels_one = 0
            labels_zero = 0

            for batch_idx, dataset in enumerate(trainloader):
                data = dataset['input_ids']
                target = dataset['label']

                attention_mask = dataset['attention_mask']
                data, target, mask = data.to(device), target.to(device), attention_mask.to(device)
                distilbert_output = model.distill_bert(data, mask)
                hidden_state = distilbert_output[0] # Batch x Tokens x Feature
                features = hidden_state[:, 0] # Batch x Feature -> by taking 1 token, alternative, 

                target_list = target.tolist()
                one = target_list.count(1)
                zero = target_list.count(0)
                labels_one += one
                labels_zero += zero

                for i,t in enumerate(target):
                    class_sums[t]+=features[i].data.squeeze()

            class_sums[0] = class_sums[0]/labels_zero
            class_sums[1] = class_sums[1]/labels_one
            class_means = class_sums

            model.out.weight.data = torch.nn.functional.normalize(class_means)
        elif cfg.trainer.model == "distillbert" and cfg.dataset.name == "ag_news":
            class_sums = torch.zeros((4,768)).to(device)

            labels_zero = 0
            labels_one = 0
            labels_two = 0
            labels_three = 0

            for batch_idx, dataset in enumerate(trainloader):
                data = dataset['input_ids']
                target = dataset['label']

                attention_mask = dataset['attention_mask']
                data, target, mask = data.to(device), target.to(device), attention_mask.to(device)
                distilbert_output = model.distill_bert(data, mask)
                hidden_state = distilbert_output[0] # Batch x Tokens x Feature
                features = hidden_state[:, 0] # Batch x Feature -> by taking 1 token, alternative, 

                target_list = target.tolist()
                zero = target_list.count(0)
                one = target_list.count(1)
                two = target_list.count(2)
                three = target_list.count(3)
                labels_one += one
                labels_zero += zero
                labels_two += two
                labels_three += three

                for i,t in enumerate(target):
                    class_sums[t]+=features[i].data.squeeze()

            class_sums[0] = class_sums[0]/labels_zero
            class_sums[1] = class_sums[1]/labels_one
            class_sums[2] = class_sums[2]/labels_two
            class_sums[3] = class_sums[3]/labels_three
            class_means = class_sums

            model.out.weight.data = torch.nn.functional.normalize(class_means)
        else:
            class_sums = torch.zeros((cfg.dataset.num_classes, 512)).to(device)
            class_count = torch.zeros(cfg.dataset.num_classes).to(device)
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.to(device), target.to(device)
                if cfg.trainer.model == 'resnet':
                    features = torch.nn.Sequential(*list(global_model.model.children())[:-1])(data)
                else:
                    features = global_model.model.features(data)
                features = F.adaptive_avg_pool2d(features, 1).squeeze()

                for i, t in enumerate(target):
                    class_sums[t] += features[i].data.squeeze()
                    class_count[t] += 1
            class_means = torch.div(class_sums, torch.reshape(class_count, (-1, 1)))
            if cfg.trainer.model == 'resnet':
                global_model.model.fc.weight.data = torch.nn.functional.normalize(class_means)
            else:
                global_model.model.classifier[2].weight.data = torch.nn.functional.normalize(class_means)

    '''# save init checkpoint
    cp = global_model().model.state_dict()
    torch.save(cp, f'model_checkpoints/initialization_checkpoint.pt')'''

    # get test accuracy after NCM init (no training)
    test_accuracy, test_loss = inference(global_model.model, testloader, device)
    if cfg.wandb.activate:
        wandb.log({"init_acc": test_accuracy})
    print(f'\nInit Testing (no training)--> loss: {test_loss} accuracy: {test_accuracy}\n')

    print(global_model.model)
    # set optimizer
    # TODO: figure out how to set the optimizers using definition in the config
    if cfg.set_optimizer == 'sgd':
        opt = LocalOptimizerSGDConfig(lr=cfg.trainer.client.optimizer.lr)
    elif cfg.set_optimizer == 'adam':
        # note: ADAM automatically sets weight decay 1e-5
        opt = LocalOptimizerAdamConfig(lr=cfg.trainer.client.optimizer.lr)
    else:
        raise Exception(f"value {cfg.set_optimizer._base_} specified for cfg.set_optimizer._base_ is invalid")

    # set server optimizer based on fl_algorithm
    if cfg.trainer.fl_algorithm.lower() == 'fedavgm':
        server_opt = FedAvgWithLROptimizerConfig(lr=cfg.trainer.server.server_optimizer.lr,
                                                 momentum=cfg.trainer.server.server_optimizer.momentum)
    elif cfg.trainer.fl_algorithm.lower() == 'fedadam':
        server_opt = FedAdamOptimizerConfig(lr=cfg.trainer.server.server_optimizer.lr,
                   weight_decay=cfg.trainer.server.server_optimizer.weight_decay,
                   beta1=cfg.trainer.server.server_optimizer.beta1,
                   beta2=cfg.trainer.server.server_optimizer.beta2,
                   eps=cfg.trainer.server.server_optimizer.eps,)
    else:
        server_opt = FedAvgOptimizerConfig()

    # this version of trainer used in all 4 algorithm cases
    trainer = SyncTrainer(
            model=global_model,
            cuda_enabled=False,
            **OmegaConf.structured(SyncTrainerConfig(
                epochs=cfg.trainer.epochs,
                do_eval=cfg.trainer.do_eval,
                always_keep_trained_model=False,
                train_metrics_reported_per_epoch=cfg.trainer.train_metrics_reported_per_epoch,
                eval_epoch_frequency=1,
                report_train_metrics=cfg.trainer.report_train_metrics,
                report_train_metrics_after_aggregation=cfg.trainer.report_train_metrics_after_aggregation,
                client=ClientConfig(
                    epochs=cfg.trainer.client.epochs,
                    optimizer=opt,
                    lr_scheduler=cfg.trainer.client.lr_scheduler,
                    shuffle_batch_order=False,
                ),
                channel=FLChannelConfig(),
                server=SyncServerConfig(
                    active_user_selector=UniformlyRandomActiveUserSelectorConfig(),
                    server_optimizer=server_opt
                ),
                users_per_round=cfg.trainer.users_per_round,
                dropout_rate=cfg.trainer.dropout_rate,
                wandb=cfg.wandb,
                wsm=cfg.trainer.wsm,
            )),
        )

    print(f"\nClients in total: {data_provider.num_train_users()}")

    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT], wandb_dict=cfg.wandb)

    print("Before training")
    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1,
    )

    # get test accuracy after phase1 training
    test_accuracy, test_loss = inference(final_model.model, testloader, device)
    print(f'\nFinal Phase1 Testing--> loss: {test_loss} accuracy: {test_accuracy}\n')

    if cfg.wandb.activate:
        wandb.log({"final_acc": test_accuracy})

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )

    '''# save final checkpoint
    cp = global_model().model.state_dict()
    torch.save(cp, f'model_checkpoints/final_checkpoint.pt')'''

    trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )

def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    main(cfg)


if __name__ == "__main__":
    cfg = fl_config_from_json(json_config)
    # update with commandline args if they are passed, otherwise defaults to hardcoded cfg file vars
    set_cfg_from_cl(cfg)
    print(cfg)
    validata_dataset_params(cfg)
    run(cfg)