"""An example of code submission for the AutoML Decathlon challenge.

It implements 3 compulsory methods ('__init__', 'train' and 'test') and
an attribute 'done_training' for indicating if the model will not proceed more
training due to convergence or limited time budget.

To create a valid submission, zip model.py and metadata together with other necessary files
such as tasks_to_run.yaml, Python modules/packages, pre-trained weights, etc. The final zip file
should not exceed 300MB.
"""

import datetime
import logging
from typing import Tuple
import numpy as np
import os
import sys
import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)
import xgboost as xgb

# seeding randomness for reproducibility
np.random.seed(42)
torch.manual_seed(1)


def merge_batches(dataloader: DataLoader, is_single_label:bool):    
    x_batches = []
    y_batches = []
    for x,y in dataloader:
        x = x.detach().numpy()
        x = x.reshape(x.shape[0], -1)
        x_batches.append(x)
        
        y = y.detach().numpy()
        if len(y.shape)>2:
            y = y.reshape(y.shape[0], -1)
        
        if is_single_label: 
            # for the multi-class, single-label tasks, we need to change the ohe encoding to raw labels for input to training
            y = np.argmax(y, axis=1)
            
        y_batches.append(y)
    
    x_matrix = np.concatenate(x_batches, axis=0)
    y_matrix = np.concatenate(y_batches, axis=0)
    
    return x_matrix, y_matrix


def get_early_stopping_rounds(num_rows_train, min_patience=20, max_patience=300, min_rows=10000):

    modifier = 1 if num_rows_train <= min_rows else min_rows / num_rows_train
    simple_early_stopping_rounds = max(
        round(modifier * max_patience),
        min_patience,
    )
    return simple_early_stopping_rounds


def get_xgb_model(task_type:str, output_size: int, num_rows: int, random_state=None):
    # Adapted from NB360 xgboost example
    early_stopping_rounds = get_early_stopping_rounds(num_rows_train=num_rows)
    # Common model params
    model_params = {
        "max_depth": 6,
        "min_child_weight": 1,
        "eta": 0.1,
        "n_jobs": -1,
        "gpu_id": 0,
        "early_stopping_rounds": early_stopping_rounds,
        # "tree_method": "gpu_hist",
        "subsample": 1,
        # "sampling_method": "gradient_based",
        "gamma": 0.01,
        "colsample_bytree": 1,
        "reg_alpha": 0,
        "reg_lambda": 0
    }
    if random_state:
        model_params["random_state"]=random_state
    
    # Cases
    if task_type=="single-label":
        if output_size>2: # multi-class
            model_params = {
                "objective": "multi:softmax",
                "eval_metric": "merror",
                "num_class": output_size,
                **model_params,
            }
        else:
            model_params = { # binary
                "objective": "binary:logistic",
                **model_params,
            }
        model = xgb.XGBClassifier(**model_params)
    elif task_type=="multi-label":
        model_params = {
            **model_params,
        }
        model = xgb.XGBClassifier(**model_params)
    elif task_type=="continuous":
        model_params = {
            **model_params,
        }
        model = xgb.XGBRegressor(**model_params)
    else: 
        raise NotImplementedError
        
    return model

def get_lgb_model(task_type:str, output_size: int, num_rows: int, random_state=None):
    # Adapted from NB360 xgboost example
    early_stopping_rounds = get_early_stopping_rounds(num_rows_train=num_rows)
    # Common model params
    model_params = {
        "min_data_in_leaf": 20,
        "feature_fraction": 1,
        "learning_rate": 0.05,
        "n_jobs": -1,
        "gpu_id": 0,
        "early_stopping_rounds": early_stopping_rounds,
        # "tree_method": "gpu_hist",
        "num_leaves": 31,
        "num_rounds": 10000,
        # "sampling_method": "gradient_based",
        "gamma": 0.01,
        "extra_trees": False,
        # "reg_alpha": 0,
        # "reg_lambda": 0
    }
    if random_state:
        model_params["random_state"]=random_state
    
    # Cases
    if task_type=="single-label":
        if output_size>2: # multi-class
            model_params = {
                "objective": "multiclass",
                "eval_metric": "multi_logloss",
                "num_class": output_size,
                **model_params,
            }
        else:
            model_params = { # binary
                "objective": "binary",
                "eval_metric": 'binary_logloss',
                **model_params,
            }
        model = LGBMClassifier(**model_params)
    elif task_type=="continuous":
        model_params = {
            **model_params,
        }
        model = LGBMRegressor(**model_params)
    else: 
        raise NotImplementedError
        
    return model


def get_rf_model(task_type:str, output_size: int, num_rows: int, random_state=None):
    # Common model params
    model_params = {
        "criterion": "gini",
        "learning_rate": 0.05,
        "n_estimators": 300,
        "bootstrap":True,
        "n_jobs": -1,
        "class_weight": "balanced_subsample"
    }
    if random_state:
        model_params["random_state"]=random_state
    
    # Cases
    if task_type=="single-label":
        # if output_size>2: # multi-class
        #     model_params.update({"n_estimators": 8})
        # else:
        #     model_params = { # binary
        #         "objective": "binary",
        #         "eval_metric": 'binary_logloss',
        #         **model_params,
        #     }
        model = RandomForestClassifier(**model_params)
    elif task_type=="continuous":
        # model_params = {
        #     **model_params,
        # }
        model = RandomForestRegressor(**model_params)
    else:
        raise NotImplementedError
        
    return model


def get_traditional_model(task_type:str, output_size: int, num_rows: Tuple, name: str, random_state=None):
    kwargs = dict(task_type=task_type, output_size=output_size, random_state=random_state, num_rows=num_rows)
    if name == 'xgb':
        return get_xgb_model(**kwargs)
    elif name == 'lgb':
        return get_lgb_model(**kwargs)
    elif name == "rf":
        return get_rf_model(**kwargs)
    else:
        raise NotImplementedError
    

class TraditionalModel:
    def __init__(self, metadata, name):
        '''
        The initalization procedure for your method given the metadata of the task
        '''
        """
        Args:
          metadata: an DecathlonMetadata object. Its definition can be found in
              ingestion/dev_datasets.py
        """
        self.name = name
        # Attribute necessary for ingestion program to stop evaluation process
        self.done_training = False
        self.metadata_ = metadata
        self.task = self.metadata_.get_dataset_name()
        self.task_type = self.metadata_.get_task_type()
        
        # Getting details of the data from meta data
        # Product of output dimensions in case of multi-dimensional outputs...
        self.output_dim = math.prod(self.metadata_.get_output_shape())

        self.num_examples_train = self.metadata_.size()

        row_count, col_count = self.metadata_.get_tensor_shape()[2:4]
        channel = self.metadata_.get_tensor_shape()[1]
        sequence_size = self.metadata_.get_tensor_shape()[0]

        self.num_train = self.metadata_.size()
        self.num_test = self.metadata_.get_output_shape()

        # Getting the device available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "Device Found = ", self.device, "\nMoving Model and Data into the device..."
        )
        if name == "xgb":
            assert torch.cuda.is_available() # force xgboost on gpu
        self.input_shape = (channel, sequence_size, row_count, col_count)
        print("\n\nINPUT SHAPE = ", self.input_shape)

        # Creating xgb model
        self.model = get_traditional_model(self.task_type, self.output_dim, self.name)
        
        # Attributes for managing time budget
        # Cumulated number of training steps
        self.birthday = time.time()
        self.total_train_time = 0
        self.total_test_time = 0

#         # no of examples at each step/batch
        self.train_batch_size = 64
        self.test_batch_size = 64

    def get_dataloader(self, dataset, batch_size, split):
        """Get the PyTorch dataloader. Do not modify this method.
        Args:
          dataset:
          batch_size : batch_size for training set
        Return:
          dataloader: PyTorch Dataloader
        """
        if split == "train":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=True,
                drop_last=False,
                collate_fn=dataset.collate_fn,
            )
        elif split == "test":
            dataloader = DataLoader(
                dataset,
                dataset.required_batch_size or batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
            )
        return dataloader

    def train(self, dataset, val_dataset=None, val_metadata=None, remaining_time_budget=None):
        '''
        The training procedure of your method given training data, validation data (which is only directly provided in certain tasks, otherwise you are free to create your own validation strategies), and remaining time budget for training.
        '''
        
        """Train this algorithm on the Pytorch dataset.
        ****************************************************************************
        ****************************************************************************
        Args:
          dataset: a `DecathlonDataset` object. Each of its examples is of the form
                (example, labels)
              where `example` is a dense 4-D Tensor of shape
                (sequence_size, row_count, col_count, num_channels)
              and `labels` is a 1-D or 2-D Tensor
          val_dataset: a 'DecathlonDataset' object. Is not 'None' if a pre-split validation set is provided, in which case you should use it for any validation purposes. Otherwise, you are free to create your own validation split(s) as desired.
          
          val_metadata: a 'DecathlonMetadata' object, corresponding to 'val_dataset'.
          remaining_time_budget: time remaining to execute train(). The method
              should keep track of its execution time to avoid exceeding its time
              budget. If remaining_time_budget is None, no time budget is imposed.
              
          remaining_time_budget: the time budget constraint for the task, which may influence the training procedure.
        """

        # If PyTorch dataloader for training set doen't already exists, get the train dataloader
        if not hasattr(self, "trainloader"):
            self.trainloader = self.get_dataloader(
                dataset,
                self.train_batch_size,
                "train",
            )

        train_start = time.time()

        # Training (no loop)
        x_train, y_train = merge_batches(self.trainloader, (self.task_type=="single-label") )
        print(x_train.shape, y_train.shape)
        if val_dataset:
            valloader = self.get_dataloader(val_dataset, self.test_batch_size, "test")
            x_valid, y_valid = merge_batches(valloader, (self.task_type=="single-label") )
        else:
            random_state=None # can set this for reproducibility if desired
            x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, random_state=random_state)
        
        fit_params = {"verbose":True}

        ##############################
        if self.name in ["xgb", "lgb"]:
            fit_params.update(dict(eval_set=[(x_valid, y_valid)]))

        self.model.fit(
            x_train,
            y_train,
            **fit_params,
        )
        ##############################       
        train_end = time.time()


        train_duration = train_end - train_start
        self.total_train_time += train_duration
        logger.info(
            "{:.2f} sec used for {}. ".format(
                train_duration, self.name
            )
            + "Total time used for training: {:.2f} sec. ".format(
                self.total_train_time
            )
        )


    def test(self, dataset, remaining_time_budget=None):
        """Test this algorithm on the Pytorch dataloader.
        Args:
          Same as that of `train` method, except that the `labels` will be empty.
        Returns:
          predictions: A `numpy.ndarray` matrix of shape (sample_count, output_dim).
              here `sample_count` is the number of examples in this dataset as test
              set and `output_dim` is the number of labels to be predicted. The
              values should be binary or in the interval [0,1].
          remaining_time_budget: the remaining time budget left for testing, post-training 
        """

        test_begin = time.time()

        logger.info("Begin testing...")

        if not hasattr(self, "testloader"):
            self.testloader = self.get_dataloader(
                dataset,
                self.test_batch_size,
                "test",
            )
        
        x_test, _ = merge_batches(self.testloader, (self.task_type=="single-label") )
        
        # get test predictions from the model
        predictions = self.model.predict(x_test)
        # If the task is multi-class single label, the output will be in raw labels; we need to convert to ohe for passing back to ingestion
        if (self.task_type=="single-label"):
            n = self.metadata_.get_output_shape()[0]
            predictions = np.eye(n)[predictions.astype(int)]
        
        test_end = time.time()
        # Update some variables for time management
        test_duration = test_end - test_begin
        self.total_test_time += test_duration

        logger.info(
            "[+] Successfully made one prediction. {:.2f} sec used. ".format(
                test_duration
            )
            + "Total time used for testing: {:.2f} sec. ".format(self.total_test_time)
        )
        return predictions

    ##############################################################################
    #### Above 3 methods (__init__, train, test) should always be implemented ####
    ##############################################################################


def get_logger(verbosity_level):
    """Set logging format to something like:
    2019-04-25 12:52:51,924 INFO model.py: <message>
    """
    logger = logging.getLogger(__file__)
    logging_level = getattr(logging, verbosity_level)
    logger.setLevel(logging_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(filename)s: %(message)s"
    )
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging_level)
    stdout_handler.setFormatter(formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)
    logger.addHandler(stderr_handler)
    logger.propagate = False
    return logger


logger = get_logger("INFO")
