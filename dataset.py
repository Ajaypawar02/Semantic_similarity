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
from args import args
warnings.filterwarnings("ignore")
import gc
gc.enable()




class Data_class(Dataset):
    def __init__(self, df,args, inference_only=False):
        super().__init__()
        
        self.df = df      
#         df["airline_sentiment"] = df["airline_sentiment"].apply(lambda x : add_sentiment(x))
        self.inference_only = inference_only
#         self.text = df.text.tolist()
        
        if not self.inference_only:
            self.target = torch.tensor(df.label.values, dtype=torch.float)     
            
        self.sen_1 = df["text"].tolist()
        self.sen_2 = df["reason"].tolist()
    
        self.encoded = args.tokenizer.encode_plus(str(self.sen_1), 
                      str(self.sen_2),
                      padding = "max_length", 
                      add_special_tokens = True, 
                      truncation = True, 
                      return_attention_mask = True)      
 

    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, index):    
        
        encoded = args.tokenizer.encode_plus(self.sen_1[index], 
                      self.sen_2[index],
                      padding = "max_length", 
                      add_special_tokens = True,
                      max_length = args.MAX_LEN,
                      truncation = True, 
                      return_attention_mask = True)   
        input_ids = torch.tensor(encoded["input_ids"])
        attention_mask = torch.tensor(encoded["attention_mask"])
        token_type_ids = torch.tensor(encoded["token_type_ids"])
        
        
        if self.inference_only:
            return {
                "input_ids" : input_ids, 
                "attention_mask" : attention_mask, 
                "token_type_ids" : token_type_ids
            }           
        else:
            target = self.target[index]
            return {
                "input_ids" : input_ids, 
                "attention_mask" : attention_mask, 
                "token_type_ids" : token_type_ids,
                "target" : target
            }