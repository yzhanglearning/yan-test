## an example for training ##

import torch
from torch.nn import functional as F
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import BertTokenizer






model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True)
model.train()
optimizer = AdamW(model.parameters(), lr=1e-5)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_batch = ["I love Pixar.", "I don't care for Pixar."]

encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']

labels = torch.tensor([1,0])

#labels = torch.tensor([1,0]).unsequeeze(0)
#outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#loss = outputs.loss


model.to('cuda')
input_ids = input_ids.to('cuda')
attention_mask = attention_mask.to('cuda')
labels = labels.to('cuda')

outputs = model(input_ids, attention_mask=attention_mask)
loss = F.cross_entropy(outputs.logits, labels)
loss.backward()
optimizer.step()

print('the current loss is {}'.format(loss))




