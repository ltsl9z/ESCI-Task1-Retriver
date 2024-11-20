# ESCI-Task1-Retriver
Fine tuning pretrained model on ESCI task 1 with English locale product

## Dataset Selected
- dataset link : https://huggingface.co/datasets/alvations/esci-data-task1
- description : We introduce the “Shopping Queries Data Set”, a large dataset of difficult search queries, released with the aim of fostering research in the area of semantic matching of queries and products. For each query, the dataset provides a list of up to 40 potentially relevant results, together with ESCI relevance judgements (Exact, Substitute, Complement, Irrelevant) indicating the relevance of the product to the query. Task 1 of this bencmark dataset is ` Given a user specified query and a list of matched products, the goal of this task is to rank the products so that the relevant products are ranked above the non-relevant ones.`
- Filter : only focused on English locale of the dataset, as the JP, ES locale data points require a different multilingual model to train.
- task 1 english local dataset summary :

|       | Total | Total | Total | Train | Train | Train | Test | Test | Test |
| ------------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- | ---------- | ------------- | ---------- |
| Language      | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth | \# Queries | \# Judgements | Avg. Depth |
| English (US)  | 97,345     | 1,818,825     | 18.68      | 74,888     | 1,393,063     | 18.60      | 22,458     | 425,762       | 18.96      |

## Requirements
```
numpy==1.26.4
pandas==2.2.2
transformers==4.46.2
scikit-learn==1.5.2
sentence-transformers==3.2.1
pytrec_eval==0.5
tqdm==4.66.6
PyYAML==6.0.2
```
### Setup Dependencies and Environment
```
make create_env
make install_deps
source .venv/bin/activate
```

## Fine Tunining and Evaluation Pipeline Execution
### Fine tuning
To kick off the fine tunining pipeline, the execution scrupt will load the model hyperparameter config files and dependent training script to execute. After the fine tuning , the model will be automatically evaluated with the test set.
``` bash
start_experiment.sh
```
