import argparse
import os
import warnings
from exp.experiment import Exp_main
from utils.util import mkdir_p
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Time series anomaly detection Experiments")
    # Data Configs
    args.add_argument("--dataset", type=str, default="Discords_Data", help="dataset name")
    args.add_argument("--datafile", type=str, default="noisy_sine.txt", help="data file name")
    args.add_argument("--result_path", type=str, default="./results", help="result saving path")
    args.add_argument("--data_path", type=str, default="./data", help="data folder")
    args.add_argument("--checkpoint", type=str, default="./checkpoints", help="model saving path")
    args.add_argument("--annotation", default=[5,10,20,30], help="the anomaly subsequence index")
    # Data processing, like sliding window(time segment), standardization, etc
    args.add_argument("--window_size", type=int, default=300, help="window/time series segment size")
    args.add_argument("--stride", type=int, default=20, help="stride step")
    args.add_argument("--hist_size", type=float, default=0.7, help="the history length of one segment")
    args.add_argument("--scaler", type=str, default="Standard", help="what scaler to use, options: ['Standard', 'MinMax']")
    args.add_argument("--split", type=list, default=[0.6, 0.2, 0.2], help="Train/Test split ratio")
    
    # Model Configs
    ## Embedding methods
    
    ## Detection methods

    ## END2END methods
    args.add_argument("--batch_size", type=int, default=32, help="batch size of deep models")
    args.add_argument("--z_dim", type=int, default=32, help="latent space dimension for AE")
    args.add_argument("--n_epochs", type=int, default=20, help="Training epochs")
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--input_c", type=int, default=1)
    args.add_argument("--output_c", type=int, default=1)

    ## AnoTran
    args.add_argument("--k", type=int, default=3)

    ## IDK methods
    args.add_argument("--psi1", type=int, default=4, help="sampling size of first IDK")
    args.add_argument("--psi2", type=int, default=4, help="sampling size of second IDK")
    args.add_argument("--t", type=int, default=100, help="sampling time of IDK")

    # optimization Configs(for embedding, end-to-end methods)
    
    # Others
    args.add_argument("--save_embedding", type=bool, default=True, help="whether save embedding like for visualization")
    

    Configs = args.parse_args()
    Configs.hist_size = int(Configs.hist_size * Configs.window_size)
    if isinstance(Configs.annotation, str):
        Configs.annotation = pd.read_csv(os.path.join(Configs.data_path, Configs.annotation), header=None).to_numpy().flatten().tolist()
    mkdir_p(os.path.join(Configs.result_path, Configs.dataset))
    mkdir_p(Configs.checkpoint)

    exp = Exp_main(Configs)
    exp.run()
