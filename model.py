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




class SEN_Model(nn.Module):
    def __init__(self):
        super().__init__()

        config = AutoConfig.from_pretrained(args.BERT_PATH)
        config.update({"output_hidden_states":True, 
                       "layer_norm_eps": 1e-7})                       
        self.layer_start = 9
        self.bert = AutoModel.from_pretrained(args.BERT_PATH, config=config)  

        self.attention = nn.Sequential(            
            nn.Linear(768, 512),            
            nn.Tanh(),                       
            nn.Linear(512, 1),
            nn.Softmax(dim=1)
        )        

        self.linear = nn.Linear(768, 1)
#         self.softmax = nn.Softmax(dim = -1)
        

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        #print(outputs)
        #all_hidden_state = outputs.hidden_states[-1]
       # weighted_pooling_embeddings = self.pooler(all_hidden_state)
#         print(outputs.hidden_states[-1].shape)
        
        weights = self.attention(outputs.hidden_states[-1])
        #[batch_size, max_len, hidden_states]
#         print(weights.shape)
        
       
        context_vector = torch.sum(weights *outputs.hidden_states[-1] , dim=1) 
#         print((weights *outputs.hidden_states[-1]).shape)
#         print(context_vector.shape)
        
        return self.linear(context_vector)


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
    
    
    
    
    
    