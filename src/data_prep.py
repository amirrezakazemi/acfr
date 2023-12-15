import numpy as np
import pandas as pd
import os
import math
from sklearn.model_selection import train_test_split
import pickle
import csv
import logging
import sys


# Configure the logging module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

def make_count_matrix(data):
    count_data = np.zeros((5000, 3477))
    for j in range(data.shape[0]):
        count_data[data[j, 0] - 1, data[j, 1] - 1] = data[j, 2]
    return count_data


def normalize_data(data):
    x = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-10)
    for i in range(x.shape[0]):
        x[i] = x[i] / np.linalg.norm(x[i])
    return x


def get_v(run_n):
    np.random.seed(run_n)
    v1p, v2p, v3p = np.random.normal(0, 1, NUMBER_OF_FEATURES), np.random.normal(0, 1,
                                                                                    NUMBER_OF_FEATURES), np.random.normal(
        0, 1, NUMBER_OF_FEATURES)
    v1, v2, v3 = v1p / np.linalg.norm(v1p, 2), v2p / np.linalg.norm(v2p, 2), v3p / np.linalg.norm(v3p, 2)
    return v1, v2, v3


def compute_beta(alpha, optimal_dosage):
    if (optimal_dosage <= 0.001 or optimal_dosage >= 1.0):
        beta = 1.0
    else:
        beta = (alpha - 1.0) / float(optimal_dosage) + (2.0 - alpha)
    return beta

def get_t(x, v2, v3, selection_bias):

    optimal_dosage = np.dot(v3, x) / (2.0 * np.dot(v2, x))
    alpha = selection_bias
    t = np.random.beta(alpha, compute_beta(alpha, optimal_dosage))
    if optimal_dosage <= 0.001:
        t = 1 - t
    return t


def get_y(t, x, v1, v2, v3):

    y = 10.0 * (np.dot(v1, x) + np.sin(
        np.pi * (np.dot(v2, x) / np.dot(v3, x)) * t))

    return y + np.random.normal(0, 0.2)

def po(test_data, v1, v2, v3, n_grids=2**6+1):
    n_test = test_data.shape[0]
    ts = np.linspace(0.01, 1, n_grids)
    t_grids = np.zeros((n_test + 1, n_grids))
    t_grids[0, :] = ts.squeeze()
    t_grid = np.zeros((2, n_grids))
    t_grid[0, :] = ts.squeeze()

    for j in range(n_grids):
        t = ts[j]
        psi = 0
        for i in range(n_test):
            x = test_data[i, 0:NUMBER_OF_FEATURES]
            y_hat_ij = get_y(t, x, v1, v2, v3)
            t_grids[i + 1, j] = y_hat_ij
            psi += y_hat_ij
        t_grid[1, j] = (psi / n_test)
    return t_grid, t_grids

def make_continuous(data, v1, v2, v3, selection_bias):
    tmp = np.zeros((data.shape[0], 2))
    continuous_data = np.append(data, tmp, axis=1)
    for row in range(data.shape[0]):
        ## adding continuous treatment
        t = get_t(data[row, :], v2, v3, selection_bias)
        continuous_data[row, data.shape[1]] = t

        ## adding outcome
        y = get_y(t, data[row, :], v1, v2, v3)
        continuous_data[row, data.shape[1] + 1] = y

    return continuous_data


def create_synthetic_data(dataset="tcga", dir='dataset/tcga/', run_n=0, selection_bias=2):
    global NUMBER_OF_FEATURES
    if dataset == "tcga":

        x_dir = f'{dir}/tcga.p'
        assert os.path.exists(x_dir), logger.error(f'tcga raw dataset does not exist')
        NUMBER_OF_FEATURES = 4000
        data = pickle.load(open(x_dir, 'rb'))
        data = normalize_data(data['rnaseq'])

    elif dataset == "news":
        
        x_dir = f'{dir}/topic_doc_mean_n5000_k3477_seed_1.csv.x'
        assert os.path.exists(x_dir), logger.error(f'news raw dataset does not exist')
        NUMBER_OF_FEATURES = 3477   
        raw_data = np.loadtxt(x_dir, delimiter=",", dtype=int)[1:, :]
        data = make_count_matrix(raw_data)

    v1, v2, v3 = get_v(run_n)
    
    curr_dir = f'{dir}/{str(run_n)}'
    
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
    else:
        logger.info(f'dataset {run_n} directory exists')

    if os.path.exists(curr_dir + "/train.csv") \
        and os.path.exists(curr_dir + "/test.csv") \
        and os.path.exists(curr_dir + "/val.csv"):
        logger.info(f"Train, validation, test for tcga data {run_n} exist.")
    else:
        continuous_data = make_continuous(data, v1, v2, v3, selection_bias)

        train, test = train_test_split(continuous_data, train_size=2/3)
        train, val = train_test_split(train, train_size=0.9)

        
        np.savetxt(curr_dir + "/train.csv", train, delimiter=",")
        np.savetxt(curr_dir + "/val.csv", val, delimiter=",")
        np.savetxt(curr_dir + "/test.csv", test, delimiter=",")


    if os.path.exists(curr_dir + "/out_t_grids.csv") \
        and os.path.exists(curr_dir + "/out_t_grid.csv") \
        and os.path.exists(curr_dir + "/in_t_grids.csv") \
        and os.path.exists(curr_dir + "/in_t_grid.csv"):
        logger.info(f"Potential outcomes for tcga dataset {run_n} exist.")
    else:
        test = np.loadtxt(curr_dir + "/test.csv", delimiter=",")
        train = np.loadtxt(curr_dir + "/train.csv", delimiter=",")

        out_t_grid, out_t_grids = po(test, v1, v2, v3)
        in_t_grid, in_t_grids = po(train, v1, v2, v3)


        np.savetxt(curr_dir + "/out_t_grids.csv", out_t_grids, delimiter=",")
        np.savetxt(curr_dir + "/out_t_grid.csv", out_t_grid, delimiter=",")

        np.savetxt(curr_dir + "/in_t_grids.csv", in_t_grids, delimiter=",")
        np.savetxt(curr_dir + "/in_t_grid.csv", in_t_grid, delimiter=",")

