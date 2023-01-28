import numpy as np
import pandas as pd
import pickle

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.metrics import f1_score
# from torchcontrib.optim import SWA
from transformers import AutoTokenizer, AutoModel, AutoConfig,BertModel, BertTokenizer
# from torchcontrib.optim import SWA

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import model_selection
from tqdm import tqdm
# import seaborn as sns

from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import gc
gc.enable()


class args:
    train_path = r"C:\Users\ajayp\OneDrive\Desktop\Semantic_similarity\data\training_file.csv"
    test_path = r"C:\Users\ajayp\OneDrive\Desktop\Semantic_similarity\data\testing_file.csv"
    TOKENIZER_PATH = "bert-base-uncased"
    BERT_PATH = "bert-base-uncased"
    ROBERTA_PATH = "roberta-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    MAX_LEN = 256
    train_batch_size = 16
    valid_batch_size = 1
    epochs = 15
    model_path = r"C:\Users\ajayp\OneDrive\Desktop\Semantic_similarity\Saved_model_weights"
    folds_path = r"C:\Users\ajayp\OneDrive\Desktop\Semantic_similarity\data\train_folds.csv"
    splits  = 5


