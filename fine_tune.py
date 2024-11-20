import argparse
import pandas as pd
import numpy as np
import torch
import yaml
import sys
from pprint import pprint
import subprocess
from google.colab import files
from code_utils.dataset import ESCIDataLoader
from code_utils.custom_evaluator import CustomRerankingEvaluator
from code_utils.rerank_model import load_crossencoder, crossencoder_fine_tune, set_seed

def main() -> None:
    ## declare core input parameters
    CONFIG_PATH = './configs'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type = str, help = "model fine tune configuration file name.")
    parser.add_argument("--locale", type = str, default = 'us', help = "product locale for the query and matching items in ESCI dataset.")
    parser.add_argument("--colab_mode", type = bool, default = True, help = "whether to save the model from colab to local drive.")
    args = parser.parse_args()

    # fixed random generator
    set_seed()

    ## Loading the dataset
    print(f'-------------------Loading the ESCI dataset for task1 with product locale={args.locale}-------------------')
    esci_data_loader = ESCIDataLoader(locale = args.locale)
    train_dataset_tok, val_dataset_tok, test_dataset_tok = esci_data_loader.load_tensor_dataset()
    print(f'-------------------Dataset loading complete-------------------')

    with open(f'{CONFIG_PATH}/{args.config_name}', 'r') as file:
        config = yaml.safe_load(file)
    ## sanity check the configs
    print('-----------------------please double check the config below and decide if want to proceed with finetuning-----------------------')
    pprint(config)
    print()
    print('----------------------------------------------')
    choice = input('Press q to Quit or enter to Continue\n')
    if choice == 'q':
        sys.exit(0)
    elif choice == '':
        print('Jobs continue...')
    ## loading pretrained model
    model = load_crossencoder(config)

    ## fine tune start
    crossencoder_fine_tune(
        model = model,
        config = config,
        train_dataset = train_dataset_tok,
        eval_dataset = val_dataset_tok
    )

    ## optional : download the saved model to local in case it's removed from colab
    output_dir = config['artifacts_path']['output_dir']
    if args.colab_mode:
        process = subprocess.Popen(['zip', '-r', './fine_tuned_model.zip', f'./{output_dir}'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        files.download("fine_tuned_model.zip")



if __name__ == "__main__": 
    main()