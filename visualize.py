import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def ADRF():
    i = 0
    t_grid = "dataset/tcga/selection_bias=2/" + str(i) + "/t_grid.csv"
    t_grid = pd.read_csv(t_grid, header=None).to_numpy()
    y = t_grid[1, :]
    t = t_grid[0, :]
    sorted_idx = np.argsort(t)
    y = y[sorted_idx]
    y_hat = pd.read_csv("dataset/tcga/selection_bias=2/" + str(i) + "/psi.csv")['0']
    y_hat = y_hat.to_numpy()
    y_hat = y_hat[sorted_idx]
    t = t[sorted_idx]
    plt.plot(t, y, label="true")
    plt.plot(t, y_hat, label="prediction")
    plt.savefig("adrf.png")
    plt.show()


def IDRF():
    i = 0
    j = 140
    t_grids = "dataset/ihdp/" + str(i) + "/t_grids.csv"
    t_grids = pd.read_csv(t_grids, header=None).to_numpy()
    y = t_grids[j, :]
    t = t_grids[0, :]
    sorted_idx = np.argsort(t)
    # y_hat = pd.read_csv("dataset/ihdp/" + str(i) + "/po.csv", index_col=0).iloc[j - 1, :]
    #
    # y_hat = y_hat.to_numpy()
    # y_hat = y_hat[sorted_idx]
    t = t[sorted_idx]
    y = y[sorted_idx]
    print(t)
    print(y)
    # print(y_hat)
    plt.plot(t, y, label="true")
    #plt.plot(t, y_hat, label="prediction")
    plt.show()


def xadrf():
    i = 0
    t_grids = "dataset/tcga/" + str(i) + "/t_grids.csv"
    t_grids = pd.read_csv(t_grids, header=None).to_numpy()
    ys = t_grids[1:, :]
    x = np.arange(0, ys.shape[0])
    meany = np.mean(ys, axis=1)
    po = "dataset/tcga/" + str(i) + "/po.csv"
    po = pd.read_csv(po, index_col=0).to_numpy()
    meanyhat = np.mean(po, axis=1)
    argsort = np.argsort(meany)
    meany = meany[argsort]
    meanyhat = meanyhat[argsort]
    random = np.random.randint(x.shape[0], size=500)
    random = np.sort(random)
    plt.plot(x[random], meany[random], label="true")
    plt.plot(x[random], meanyhat[random], label="prediction")
    plt.savefig("xadrf.png")
    plt.show()






ADRF()
#IDRF()
#xadrf()
#### 1.3201880292147783, 0.5502090906148988, 0.14727748520595874, 0.1810518053462802, 0.02552242846267755, 1.0266397778161902, 0.16900295945356622, 0.03413087173181408