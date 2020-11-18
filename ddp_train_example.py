## an example for training ##

import os
import sys
import warnings

import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


#import torch
from torch.nn import functional as F
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertTokenizer



def setup(rank, world_size):
    #os.environ['SLURM_LOCALID']
    #os.environ['SLURM_NTASKS_PER_NODE']
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()




def demo_basic(rank, world_size):
    print("running basic DDP transformer example on rank {}".format(rank))
    setup(rank, world_size)

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True).to(rank)
    model.train()
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = AdamW(ddp_model.parameters(), lr=1e-5)
    #optimizer.zero_grad()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text_batch = ["I love Pixar.", "I don't care for Pixar."]
    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    labels = torch.tensor([1,0])

    #labels = torch.tensor([1,0]).unsequeeze(0)
    #outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    #loss = outputs.loss

    input_ids = input_ids.to(rank)
    attention_mask = attention_mask.to(rank)
    labels = labels.to(rank)

    outputs = ddp_model(input_ids, attention_mask=attention_mask)
    loss = F.cross_entropy(outputs.logits, labels)
    loss.backward()
    optimizer.step()

    print('the current loss is {}'.format(loss))
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    #global world_size
    #world_size=0
    run_demo(demo_basic, 4)




    
