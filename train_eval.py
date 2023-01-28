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


def train_fn(data_loader, model, optimizer,device, scheduler):
    model.train()
    
    loss_sum = 0
    
    for i, data in tqdm(enumerate(data_loader), total = len(data_loader)):
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]
        token_type_ids = data["token_type_ids"]
        target = data["target"]
        
        input_ids = input_ids.to(device, dtype = torch.long)
        attention_mask = attention_mask.to(device, dtype = torch.long)
        target = target.to(device, dtype = torch.float)
        token_type_ids = token_type_ids.to(device, dtype = torch.long)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, token_type_ids)
        
        loss = loss_fn(outputs, target)
        loss.backward()
        
        loss_sum += loss.item()
        
        optimizer.step()
        scheduler.step()
        
        
    return loss_sum/len(data_loader)
        
        
    
    
def eval_fn(data_loader, model, device):
    
    model.eval()
    
    final_targets = []
    
    final_outputs = []
    
    with torch.no_grad():
            
        for i, data in tqdm(enumerate(data_loader), total = len(data_loader)):
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            token_type_ids = data["token_type_ids"]
            target = data["target"]

            input_ids = input_ids.to(device, dtype = torch.long)
            attention_mask = attention_mask.to(device, dtype = torch.long)
            target = target.to(device, dtype = torch.float)
            token_type_ids = token_type_ids.to(device, dtype = torch.long)
            
            outputs = model(input_ids, attention_mask, token_type_ids)
            
            output = torch.sigmoid(outputs)
            
#             ans = torch.argmax(output, dim = -1)
            
            targets = target.detach().cpu().numpy().tolist()
            
            ans = output.detach().cpu().numpy().tolist()
            
            final_targets.extend(targets)
            
            final_outputs.extend(ans)
            
    return final_outputs, final_targets