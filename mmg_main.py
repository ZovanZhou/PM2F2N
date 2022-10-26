import os

# Cancel the warning info from tensorflow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import random
import pprint
import argparse
import numpy as np
import pandas as pd
from model import *
from utils import *
from tqdm import trange
import tensorflow as tf
from dataloader import DataLoader


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--weights", type=str, default="./weights/model.h5")
parser.add_argument("--save_features", type=int, default=0, choices=[0, 1])
parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])
parser.add_argument(
    "--task",
    type=str,
    default="mort_icu",
    choices=["mort_hosp", "mort_icu", "los_3", "los_7"],
)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
tf.random.set_seed(args.seed)
os.environ["TF_DETERMINISTIC_OPS"] = "1"

dataLoader = DataLoader("./data", args.task)
optimizer = tf.optimizers.Adam()
model = MultimodalGraphModel()
dev_loss = np.inf
patience_cnt = 0
patience = args.patience
best_result = {}

if args.mode == "train":
    for i in trange(args.epoch, ncols=80):
        train_one_step(model, dataLoader, optimizer)
        tmp_dev_loss = test_model_with_loss(model, dataLoader)
        if tmp_dev_loss > dev_loss:
            patience_cnt += 1
        else:
            patience_cnt = 0
            dev_loss = tmp_dev_loss
            best_result = test_model_with_metrics(model, dataLoader)
            model.save_weights(args.weights)

        if patience_cnt == patience:
            break
    pprint.pprint(best_result)

elif args.mode == "test":
    load_model(model, args.weights, dataLoader, optimizer)
    dev_result = test_model_with_metrics(model, dataLoader, dtype="dev")
    pprint.pprint(dev_result)
    test_result = test_model_with_metrics(model, dataLoader, dtype="test")
    pprint.pprint(test_result)

if args.save_features:
    features = extract_model_features(model, dataLoader, dtype="test")
    pd.to_pickle(features, f"{args.weights}.features.pkl")
