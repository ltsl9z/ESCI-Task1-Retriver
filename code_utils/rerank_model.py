from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers import evaluation
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import numpy as np
import random
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from transformers import Trainer, TrainingArguments,EarlyStoppingCallback

from sklearn.metrics import ndcg_score
from typing import List, Type, Union, Any, Dict,Tuple
from sklearn.metrics import ndcg_score


def load_crossencoder(config : Dict[str, Union[Dict, List]]) -> Any:
    """
    Load the cross encoder model from repo and instantiate.

    Args:
        config - model training config file
    Returns:
        model - instantiated model object.
    """
    # extract key params
    model_name = config['model']['name']
    num_labels = config['hyperparameters']['num_labels']
    max_length = config['hyperparameters']['max_length']
    training_blocks_config = config['training_blocks']
    print(f'-------------------Loading the {model_name} model for fine tuning-------------------')
    # define extra model loading properties
    default_activation_function = torch.nn.Identity()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'using {device.type} backend instance...')
    # define model
    model = CrossEncoder(
                model_name,
                num_labels = num_labels,
                max_length = max_length,
                default_activation_function = default_activation_function,
                device = device
            )
    # freeze encoder & pooler layers during training
    for block in training_blocks_config:
        if block['block_name'] == 'encoder':
            for param in model.model.bert.encoder.layer.parameters():
                param.requires_grad = block['training']
            if block['training']:
                print('...backpropogation enabled for layers in for %s' % block['block_name'])
            else:
                print('...backpropogation disabled for layers in for %s' % block['block_name'])
        if block['block_name'] == 'pooler':
            for param in model.model.bert.pooler.parameters():
                param.requires_grad = block['training']
            if block['training']:
                print('...backpropogation enabled for layers in for %s' % block['block_name'])
            else:
                print('...backpropogation disabled for layers in for %s' % block['block_name'])
        elif block['block_name'] == 'classifier':
            for param in model.model.classifier.parameters():
                param.requires_grad = block['training']
            if block['training']:
                print('...backpropogation enabled for layers in for %s' % block['block_name'])
            else:
                print('...backpropogation disabled for layers in for %s' % block['block_name'])

    print(f'-------------------Loading model complete-------------------')
    
    return model

def crossencoder_fine_tune(model : Any, config : Dict[str, Union[Dict, List]], train_dataset : Any, eval_dataset : Any) -> None:
    """Main training pipeline for the task 1.
    
    Args:
        model - loaded pretrained model for finetuening.
        config - model training config file.
        train_dataset - training dataset to finetune on.
        eval_dataset - validation dataset during training.
    """
    # extract key params
    batch_size = int(config['hyperparameters']['batch_size'])
    num_epochs = int(config['hyperparameters']['num_epochs'])
    lr = float(config['hyperparameters']['learning_rate'])
    warmup_ratio = float(config['hyperparameters']['warmup_ratio'])
    checkpoints_dir = config['artifacts_path']['checkpoints_dir']
    logging_dir = config['artifacts_path']['logging_dir']
    output_dir = config['artifacts_path']['output_dir']
    tensor_board_dir = config['artifacts_path']['tensor_board_dir']
    steps_per_epoch = len(train_dataset) // batch_size
    total_steps = steps_per_epoch * num_epochs
    # evaluation_steps = int(steps_per_epoch * 0.5)
    evaluation_steps = int(steps_per_epoch / 5)

    print(f'-------------------Setup configs before fine tuning-------------------')
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir = checkpoints_dir,
        evaluation_strategy = "steps",
        eval_steps = evaluation_steps,
        save_steps = evaluation_steps,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = num_epochs,
        learning_rate = lr,
        warmup_ratio = warmup_ratio,
        logging_dir = logging_dir,
        logging_steps = evaluation_steps,
        save_total_limit = 2,
        load_best_model_at_end = True,  # Load the best model at the end of training
        # metric_for_best_model = "ndcg@10",
        # greater_is_better=True,
        metric_for_best_model = "eval_loss"
    )
    # Define early stopping callback
    early_stopping = EarlyStoppingCallback(early_stopping_patience = 3, early_stopping_threshold = 0.001)
    # Initialize TensorBoard writer for logging
    writer = SummaryWriter(tensor_board_dir)
    # # set up evaluator
    # dev_evaluator = CERerankingEvaluator(dev_set, name='train-eval')

    # kick off training
    trainer = Trainer(
        model = model.model,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        tokenizer = model.tokenizer,
        # compute_metrics = compute_metrics,
        callbacks = [early_stopping],
    )
    print(f'-------------------Fine Tuning starts-------------------')
    trainer.train()
    print(f'-------------------Fine Tuning completes-------------------')
    try:
        model.save(output_dir)
    except:
        model.save_pretrained(output_dir)
        model.tokenizer.save_pretrained(output_dir)
    print(f'-------------------Saving the best model to {output_dir}-------------------')

    return

def set_seed(random_seed : int = 42) -> None:
    """fixed randomness so the experiment is reproducible."""
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_metrics(pred : Any) -> Dict[str, float]:
    """
    Compute evaluation NDCG@10.

    Args:
        pred - predicted score from the model.
    Returns:
        metric - ndcg@10
    """
    # Extract predictions and labels
    predictions = pred.predictions
    labels = pred.label_ids

    # Ensure predictions are in the correct shape (logits)
    predictions = predictions.squeeze()  # If necessary, adjust shape
    # Compute NDCG@10
    ndcg_at_10 = ndcg_score([labels], [predictions], k=10)
    
    return {"ndcg@10": ndcg_at_10}