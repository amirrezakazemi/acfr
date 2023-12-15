def get_cfg(trial, dataset):

    cfg = get_cfg_acfr(trial, dataset)
    
    return cfg


def get_cfg_acfr(trial, dataset):

    if dataset == 'news':
        input_dim = 3477
    elif dataset == 'tcga':
        input_dim = 4000


    hid_dim = trial.suggest_categorical('hid_dim', [50, 100])


    cfg = {'lr1': trial.suggest_float('lr1', 1e-5, 1e-3),
            'lr2': trial.suggest_float('lr2', 1e-5, 1e-3),
            'gamma': trial.suggest_float('gamma', 0, 1e-1),
            'enc': {'input_dim': input_dim,
                    'hidden_dims': [],
                    'output_dim': hid_dim},
            'atten': {'input_dim': hid_dim,
                    'degree': 10,
                    'hidden_dim': hid_dim,
                    'output_dim': 1},
            'disc': {'input_dim': hid_dim,
                    'hidden_dims': [],
                    'output_dim': 1}}
    return cfg