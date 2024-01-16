#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import random
import torch

import pandas as pd
import numpy as np
import torch.nn as nn

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, random_split


# In[2]:


# Set to GPU if available.
if torch.cuda.is_available():    
    device = torch.device("cuda")
    print('Using GPU:', torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print('No GPU available, using CPU instead.')


# In[3]:


# Load the dataset.
df = pd.read_json("Task 1-20231125T063955Z-001\Task 1\MaSaC_train_erc.json")

# Load the bert tokenizer.
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)

# Get the input ids (bert embedding index) and attention masks (mask of whether a token is padding).
# Note that this makes all utterances seperate examples, as bert does not support paragraphs (<SEP> only support a sentence pair).
X_input_ids = []
X_attention_mask = []

for utterances in df["utterances"]:
    for utterance in utterances:
        encoded_dict = tokenizer.encode_plus(utterance, add_special_tokens = True, max_length = 160, pad_to_max_length = True,
                                             return_attention_mask = True, return_tensors = 'pt', truncation = True)
        X_input_ids.append(encoded_dict['input_ids'])
        X_attention_mask.append(encoded_dict['attention_mask'])

# Convert emotions to labels.
label_to_index = {'contempt':0, 'anger':1, 'surprise':2, 'fear':3, 'disgust':4, 'sadness':5, 'joy':6, 'neutral':7}
Y = []
for emotions in df["emotions"]: 
    Y.extend([label_to_index[emotion] for emotion in emotions])

X_input_ids = torch.cat(X_input_ids, dim=0)
X_attention_mask = torch.cat(X_attention_mask, dim=0)
Y = torch.tensor(Y)


# In[4]:


# Prepare the dataset and data loaders.
dataset = TensorDataset(X_input_ids, X_attention_mask, Y)

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
batch_size = 32

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
validation_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)


# In[5]:


# Prepare the model. None of pretrained bert used hinglish, this is the closest.
model = BertForSequenceClassification.from_pretrained(
    "bert-base-multilingual-uncased", num_labels = 8, output_attentions = False, output_hidden_states = False,
)
model.cuda()


# In[6]:


# Prepare the optimizer and learning rate scheduler.
epochs = 4
total_steps = len(train_dataloader) * epochs
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = total_steps)


# In[7]:


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# In[8]:


seed_val = 10086
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# Training loop.
for epoch_i in range(0, epochs):
    print()
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Running Training...')

    total_train_loss = 0
    model.train()

    for step, batch in enumerate(train_dataloader):
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        model.zero_grad()

        result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)
        loss = result.loss
        logits = result.logits
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_dataloader)
    print()
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    # After each epoch, run validation once.
    print()
    print("Running Validation...")

    model.eval()

    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    val_dataloader = validation_dataloader
    # Change this to True to evaluate using training set.
    validate_on_training_set = False
    if validate_on_training_set:
        val_dataloader = train_dataloader

    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            result = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels, return_dict=True)

        loss = result.loss
        logits = result.logits
        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        total_eval_accuracy += flat_accuracy(logits, label_ids)

    avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    avg_val_loss = total_eval_loss / len(val_dataloader)
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

print()
print("Training complete!")

# Using dev set, accuracy is ~49%, which is similar to predict everything as neutral.
# Using training set, accuracy is ~62%, which means it is already overfitting and remembering examples.


# In[ ]:





# In[ ]:




