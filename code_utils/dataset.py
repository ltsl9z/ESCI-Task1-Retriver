import pandas as pd
from datasets import load_dataset, Dataset
from transformers import  AutoTokenizer
from typing import List, Type, Union, Any, Dict,Tuple

class ESCIDataLoader(object):
    """Module designed to reload the ESCI task 1 dataset."""
    def __init__(self, locale : str, data_path : str = 'alvations/esci-data-task1', tokenizer_path : str = 'cross-encoder/ms-marco-MiniLM-L-12-v2') -> None:
        self.locale = locale
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.esci_label2gain = self._esci_label2gain
        self.tokenizer = self._tokenizer
        self.col_query = "query"
        self.col_product_title = "product_title"
        self.col_label = 'esci_label'
        self.col_locale = 'product_locale'

    def load_tensor_dataset(self) -> Tuple[Any,Any,Any]:
        """Load the train, valid, test dataset from the host, perform preprocessing and convert to vectors using pretrained tokenizer."""
        train_dataset,val_dataset,test_dataset = self.load_preprocess_dataset()

        # vectorization
        train_dataset_tok = train_dataset.map(self.tokenize_function)
        train_dataset_tok.set_format("torch")
        train_dataset_tok = train_dataset_tok.remove_columns([self.col_query, self.col_product_title])
        val_dataset_tok = val_dataset.map(self.tokenize_function)
        val_dataset_tok.set_format("torch")
        val_dataset_tok = val_dataset_tok.remove_columns([self.col_query, self.col_product_title])
        test_dataset_tok = test_dataset.map(self.tokenize_function)
        test_dataset_tok.set_format("torch")
        test_dataset_tok = test_dataset_tok.remove_columns([self.col_query, self.col_product_title])

        return train_dataset_tok,val_dataset_tok,test_dataset_tok

    def load_preprocess_dataset(self) -> Tuple[Any,Any,Any]:
        """Load the train, valid, test dataset from the host and perform preprocessing."""
        # load the entire dataset
        try:
            dataset = load_dataset(self.data_path)
        except Exception as err:
            print(err)
        # preprocess the data
        train_dataset = dataset['train'].filter(lambda row: row[self.col_locale] == self.locale).select_columns([self.col_query, self.col_product_title, self.col_label])
        val_dataset = dataset['dev'].filter(lambda row: row[self.col_locale] == self.locale).select_columns([self.col_query, self.col_product_title, self.col_label])
        test_dataset = dataset['test'].filter(lambda row: row[self.col_locale] == self.locale).select_columns([self.col_query, self.col_product_title, self.col_label])

        train_dataset = train_dataset.map(self.convert_label).remove_columns(self.col_label)
        val_dataset = val_dataset.map(self.convert_label).remove_columns(self.col_label)
        test_dataset = test_dataset.map(self.convert_label).remove_columns(self.col_label)

        return train_dataset,val_dataset,test_dataset

    def to_rerank_format(self, dataset : Any) -> List[Dict[str, Union[str, List[str]]]]:
        """
        Convert the preprocessed dataset (not the tensor data one) to a format aligns with reranking evaluator.
        Args:
            dataset - preprocessed dataset before vecterization

        Returns:
            formatted_samples - same data formatted for reranking evaluator.
        """
        formatted_samples = []
        query2id = {}
        for row in dataset:
            query = row[self.col_query]
            if query not in query2id:
                query2id[query] = len(query2id)
                formatted_samples.append({'query': query, 'positive': [], 'negative': []})

            qid = query2id[query]

            if row['labels'] > 0:
                formatted_samples[qid]['positive'].append(row[self.col_product_title])
            else:
                formatted_samples[qid]['negative'].append(row[self.col_product_title])

        return formatted_samples
    
    def tokenize_function(self, examples : Any, max_length : int = 512) -> Any:
        """tokenize the inputs using predefined tokenizer
        Args:
            examples - row in transformer dataset.
            max_length - max length of output vectors after truncation.
        
        Returns:
            tokenized_input : tokenized input rows
        """
        return self.tokenizer(
            examples[self.col_query],
            examples[self.col_product_title],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )


    def convert_label(self, example : Any) -> Any:
        """Convert the esci string labels to numric matching label."""
        example['labels'] = self.esci_label2gain[example['esci_label']]
        return example

    @property
    def _tokenizer(self) -> Any:
        return AutoTokenizer.from_pretrained(self.tokenizer_path)

    @property
    def _esci_label2gain(self) -> dict[str, float]:
        """define the conversion mapping from default esci label to numeric lable between 0 and 1 for regression loss function.
        This is because of the cross encoder wil be compiled with MSE loss. 
        """
        esci_label2gain = {
            'E' : 1.0,
            'S' : 0.1,
            'C' : 0.01,
            'I' : 0.0,
        }

        return esci_label2gain