import logging
import os
import numpy as np
import statistics
import torch
import argparse
import optuna
import yaml
import warnings
from src import train
from src.data_prep import create_synthetic_data


# Configure the logging module
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)


# Set the warning filter to ignore all warnings
warnings.simplefilter("ignore")

if __name__ == "__main__":

    # Create an argument parser
    parser = argparse.ArgumentParser()

    # Define command-line arguments
    # --dataset: specifies the dataset to use, with default value 'news' and choices 'news' or 'tcga'
    parser.add_argument('--dataset', default="news", choices=['news', 'tcga'], help='Dataset to use')

    # --out_of_sample: a boolean flag that is set to True if present, otherwise False
    parser.add_argument('--out_of_sample', action='store_true')

    # --selection_bias: specifies the selection bias with a default value of 2
    parser.add_argument('--selection_bias', default=2)

    # --fine_tune_number: specifies the fine-tune number with a default value of 10
    parser.add_argument('--fine_tune_number', default=10)

    # --fine_tune: a boolean flag that is set to True if present, otherwise False
    parser.add_argument('--fine_tune', action='store_true')

    # --epoch_number: specifies the epoch number with a default value of 10
    parser.add_argument('--epoch_number', default=10)

    # --run_number: specifies the run number with a default value of 1
    parser.add_argument('--run_number', default=1)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Set random seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)



    
    MISEs = []
    models = []
    res_dir = f'results/{args.dataset}.txt'
    model_dir =f'models/{args.dataset}/'
    dataset_dir = f"data/{args.dataset}/"
    result_file = open(res_dir, 'w')

    for i in range(int(args.run_number)):

        logger.info(f'Creating synthetic data for {args.dataset}, number: {i}')
        
        create_synthetic_data(dataset=args.dataset, dir=dataset_dir, run_n=i, selection_bias=args.selection_bias)
        
        if args.fine_tune == True:
            
            logger.info(f"Fine-tuning {args.model} for {int(args.epoch_number)} trials:")
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='minimize')
            partial_run = lambda trial: train.run(trial= trial, 
                                        cfg = None,
                                        dataset_dir=dataset_dir,
                                        model_dir=model_dir,
                                        run_number = i,   
                                        epoch_n=int(args.epoch_number),
                                        dataset_type=args.dataset,
                                        out_of_sample=args.out_of_sample)

            study.optimize(partial_run, n_trials=int(args.fine_tune_number))
            best_trial = study.best_trial
            MISE = best_trial.values[0]

        else:

            logger.info(f"Using the saved yaml config")
            cfg_dir = f"configs/{args.dataset}.yaml"
            assert os.path.exists(cfg_dir), logger.error(f"The config file for this experiment does not exist.")
            with open(cfg_dir, "r") as cfg_file:
                cfg = yaml.safe_load(cfg_file)
                logger.info(cfg)

            MISE = train.run(trial= None, 
                                cfg=cfg,
                                dataset_dir=dataset_dir,
                                model_dir=model_dir,
                                run_number = i,   
                                epoch_n=int(args.epoch_number),
                                dataset_type=args.dataset,
                                out_of_sample=args.out_of_sample)
            MISE = MISE.item()

        logger.info(f"Mean Integrated Squared Error for run number {i}: {MISE}")
        print(f'Iteration {i}: MISE = {MISE}', file=result_file)
        MISEs.append(MISE)
    

    logger.info(MISEs)
    if len(MISEs) == 1:
        print(f"MISE: {MISE}", file=result_file)
    elif len(MISEs) > 1:
        print(f"Average MISE: {statistics.mean(MISEs)}", file=result_file)
        print(f"Std MISEs: {statistics.stdev(MISEs)}", file=result_file)
    
    

