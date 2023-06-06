json_config = {
    "dataset":{
        "name": "flowers",
        "num_classes": 102,
        "alpha": 0.1,
        "num_client_samples": None,
    },
    "wandb":{
        "activate": False,
        "debug": False,
        "run_name": "flowers_default",
        "offline": False,
        "project": "PretrainFlowers",
        "entity": "glegate",
        "log_freq": 1,
    },
    "trainer": {
        # there are different types of aggregator
        # fed avg doesn't require lr, while others such as fed_avg_with_lr or fed_adam do
        "_base_": "base_sync_trainer",
        "server": {
            "_base_": "base_sync_server",
            # includes all fields necessary to instantiate fedavg, fedadam or fedavgm
            "server_optimizer": {
                "_base_": "base_fed_avg_with_lr",
                "lr": 1,
                "momentum": 0.9,
                "weight_decay": 0.00001,
                "beta1": 0.9,
                "beta2": 0.999,
                "eps": 1e-8,
            },
            # type of user selection sampling
            "active_user_selector": {"_base_": "base_uniformly_random_active_user_selector"},
        },
        "client": {
            # number of client's local epoch
            "epochs": 1,
            "local_bs": 64,
             "optimizer": {
                 "_base_": "base_optimizer_adam",
                 # client's local learning rate
                "lr": 0.001,
                 # client's local momentum
                 "momentum": 0.,
             }
        },
        # model to use
        "model": 'squeezenet',
        # total clients
        "total_users": 10,
        # number of users per round for aggregation
        "users_per_round": 10,
        # total number of global epochs
        "epochs": 5,
        # frequency of reporting train metrics
        "train_metrics_reported_per_epoch": 1,
        # frequency of evaluation per epoch
        "eval_epoch_frequency": 1,
        # Whether to do evaluation and model selection based on it.
        "do_eval": True,
        # should we report train metrics after global aggregation
        "report_train_metrics_after_aggregation": True,
        "ncm_init": True,
        "wsm": False,
        "dropout_rate": 1.0,
        "last_layer": True,
        "pretrained": True,
        "algorithm": "lp",
        "fl_algorithm": "fedavg"
        },
    # specify client optimizer adam or sgd
    "set_optimizer": 'sgd',
}
