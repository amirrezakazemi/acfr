import copy
import os
import pandas as pd
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
from src.data_iterator import get_iter
from src.model import ACFR
from src.utils import get_cfg
import logging


# Configure the logging module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)


def test(model, test_data_dir, t_grids, t_grid, dataset_type):
    test_data = pd.read_csv(test_data_dir, header=None).to_numpy()
    if t_grid is not None:
        t_grid = pd.read_csv(t_grid, header=None).to_numpy()
        y = t_grid[1, :]
        y = torch.unsqueeze(torch.Tensor(y), 1)
        y_hat = torch.zeros(y.shape)

    t_grids = pd.read_csv(t_grids, header=None).to_numpy()
    t = t_grids[0, :]
    if dataset_type == 'news':
        x = test_data[:, :3477]
    elif dataset_type == 'tcga':
        x = test_data[:, :4000]
    ys = t_grids[1:, :]

    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


    t = torch.unsqueeze(torch.Tensor(t), 1).to(device)
    x = torch.Tensor(x).to(device)
    ys = torch.Tensor(ys).to(device)
    y_hats = torch.zeros((x.shape[0], t.shape[0])).to(device)
    model.to(device)
    with torch.no_grad():
        model.eval()
        for i in range(len(t)):
            tensor_i = torch.ones(x.shape[0], 1).to(device)
            tensor_i *= t[i]
            out, z = model(x, tensor_i)
            y_hats[:, i] = torch.squeeze(out, 1)
            if t_grid is not None:
                y_hat[i] = torch.mean(out)
    AMSE = None
    if t_grid is not None:
        AMSE = F.mse_loss(y, y_hat)

    MISE = torch.zeros(ys.shape[1], 1)
    for i in range(ys.shape[1]):
        MISE[i] = F.mse_loss(ys[:, i], y_hats[:, i])

    MISE = torch.mean(MISE)
    return MISE, y_hats, AMSE, y_hat


def val(model, val_iter):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with torch.no_grad():
        model.eval()
        for batch_idx, (t, x, y) in enumerate(tqdm(val_iter, desc='Batches', leave=False, disable=True)):
            x = x.flatten(start_dim=1).to(device).float()
            t = t.flatten(start_dim=0).to(device).float()
            y = y.flatten(start_dim=0).to(device).float()

            y_hat, z = model(x, t)

            loss = F.mse_loss(y, torch.squeeze(y_hat, 1))
            return loss


def train(train_iter, model, n_epoch, val_iter):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss = list()
    model.to(device)
    model.train()

    best_val = 1e10
    best_model = None
    for epoch in trange(1, n_epoch + 1, desc='Epochs', leave=True):
        

        g_loss = 0
        t_loss = 0
        for batch_idx, (t, x, y) in enumerate(tqdm(train_iter, desc='Batches', leave=False, disable=True)):
            
            x = x.flatten(start_dim=1).to(device).float()
            t = t.flatten(start_dim=0).to(device).float()
            y = y.flatten(start_dim=0).to(device).float()

            t_hat = model.forward_D(x)
            l_t = model.backward_D(t, t_hat)
            t_loss += l_t.item()

            y_hat, t_hat = model.forward_G(x, t)
            l_g = model.backward_G(t, y, t_hat, y_hat)
            loss.append(l_g.item())
            g_loss += l_g.item()
            


        val_loss = val(model, val_iter)
        logger.info(f'validation loss at epoch {epoch}: {val_loss}')
        if val_loss < best_val:
            best_model = copy.deepcopy(model)
            best_val = val_loss

    return best_model, best_val


def run(trial, cfg, dataset_dir, model_dir, run_number, epoch_n, dataset_type, out_of_sample):

    model = None
    torch.manual_seed(42)
    logger.info(f'config {cfg}')
    
    if cfg is None:
        cfg = get_cfg(trial, dataset_type)
    else:
        if 'lr1' in cfg:
            cfg['lr1'] = float(cfg['lr1'])
        if 'lr2' in cfg:
            cfg['lr2'] = float(cfg['lr2'])
        if 'gamma' in cfg:
            cfg['gamma'] = float(cfg['gamma'])
    model = ACFR(cfg)



    logger.info(f'Training model on dataset: {run_number}')

    train_data_dir = dataset_dir + str(run_number) +"/train.csv"
    val_data_dir = dataset_dir + str(run_number)  + "/val.csv"
    train_iter, val_iter = get_iter(train_data_dir, val_data_dir, dataset_type)

    ## best model for run i that is obtained in epoch j in terms of validation loss
    model, val_loss = train(train_iter, model, epoch_n, val_iter)
    logger.info(f"Best validation loss for run number {run_number}: {val_loss.item()}")
    curr_dir = f'{model_dir}{run_number}'
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
    torch.save(model.state_dict(), f'{curr_dir}/state_dict.pth')

    if out_of_sample:
        test_data_dir = dataset_dir + str(run_number) + "/test.csv"
        t_grid = dataset_dir + str(run_number) + "/out_t_grid.csv"
        t_grids = dataset_dir + str(run_number) + "/out_t_grids.csv"
    else:
        test_data_dir = dataset_dir + str(run_number) + "/train.csv"
        t_grid = dataset_dir + str(run_number) + "/in_t_grid.csv"
        t_grids = dataset_dir + str(run_number) + "/in_t_grids.csv"

    MISE, po, AMSE, psi = test(model, test_data_dir, t_grids, t_grid, dataset_type)
    return MISE


