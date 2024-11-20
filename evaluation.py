import argparse
import pandas as pd
import numpy as np
import torch
import yaml
import sys
from pprint import pprint
from google.colab import files
from code_utils.dataset import ESCIDataLoader
from code_utils.custom_evaluator import CustomRerankingEvaluator
from sentence_transformers.cross_encoder import CrossEncoder
import os

def main() -> None:
    ## declare core input parameters
    CONFIG_PATH = './configs'
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type = str, help = "model fine tune configuration file name.")
    parser.add_argument("--result_dir", type = str, help = "dir path where to store the evaluation results.")
    parser.add_argument("--locale", type = str, default = 'us', help = "product locale for the query and matching items in ESCI dataset.")
    args = parser.parse_args()


    ## Loading the dataset
    print(f'-------------------Loading the ESCI dataset for task1 with product locale={args.locale}-------------------')
    esci_data_loader = ESCIDataLoader(locale = args.locale)
    _,_,test_dataset = esci_data_loader.load_preprocess_dataset()
    print(f'-------------------Dataset loading complete-------------------')
    ## transform the test data into ranking pairs format
    test_sample = esci_data_loader.to_rerank_format(test_dataset)
    print(f'-------------------Loading the config files-------------------')
    ## Loading the config
    with open(f'{CONFIG_PATH}/{args.config_name}', 'r') as file:
        config = yaml.safe_load(file)
    model_dir = config['artifacts_path']['output_dir']
    print(f'-------------------config loading complete-------------------')

    ## Load the model and tokenizer
    print(f'-------------------Loading the pretrained model-------------------')
    model = CrossEncoder(model_dir)
    print(f'-------------------Model loading config complete-------------------')
    ## evaluate the test samples
    evaluator = CustomRerankingEvaluator(test_sample, name='test')
    if os.path.isdir(args.result_dir):
        pass
    else:
        os.mkdir(args.result_dir)
    print(f'-------------------Evaluating the test samples-------------------')
    evaluator(model, output_path = args.result_dir)
    print(f'-------------------Evaluation complete-------------------')

if __name__ == "__main__": 
    main()

