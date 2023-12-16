import argparse
import torch
import os
import pandas as pd
from data_prep_test import create_synthetic_data
from flask import Flask, jsonify
from model import ACFR

app = Flask(__name__)

def test(model, test_data_path, label_data_path):


    NUMBER_OF_FEATURES = 3477

    test_data = pd.read_csv(test_data_path, header=None).to_numpy()

    t_grids = pd.read_csv(label_data_path, header=None).to_numpy()
    t = t_grids[0, :]

    x = test_data[:, :NUMBER_OF_FEATURES]

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


    MISE = torch.zeros(ys.shape[1], 1)
    for i in range(ys.shape[1]):
        MISE[i] = torch.nn.MSELoss(ys[:, i], y_hats[:, i])

    MISE = torch.mean(MISE)
    return MISE, y_hats

@app.route('/predict', methods=['GET'])
def predict():
    
    create_synthetic_data(dataset="news", dir="data", run_n=0, selection_bias=2)

    checkpoint_path = os.getenv('MODEL_PATH', f"models/checkpoint.pth")

    data_path = os.getenv('DATA_PATH', f"data/0/test.csv")
    
    label_path = os.getenv('LABEL_PATH', f"data/0/out_t_grids.csv")

    assert os.path.exists(data_path) and os.path.exists(checkpoint_path) and os.path.exists(label_path), "The necessary files are not loaded"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(checkpoint_path, map_location=device)

    net = ACFR()
    net.load_state_dict(state_dict)

    MISE, _ = test(net, data_path, label_path)

    return jsonify({'mise': MISE.item()})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)
