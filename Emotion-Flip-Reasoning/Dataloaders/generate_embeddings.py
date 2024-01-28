import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
import pandas as pd
import nlp_utils as nu
import time
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sent2emb = {}

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")


# Load utterances
def load_utts():
    train_convs = pd.read_json("../Data/MELD_train_efr.json")["utterances"]
    test_convs = pd.read_json("../Data/MELD_test_efr.json")["utterances"]
    dev_convs = pd.read_json("../Data/MELD_dev_efr.json")["utterances"]
    convs = pd.concat([train_convs, test_convs]).to_list()
    u = [utt for conv in convs for utt in conv]  # Flatten to just utterances
    u = list(set(u))  # Remove non-unique utts
    u = [nu.preprocess_text(utt) for utt in u]
    print(f"Number of unique utterances: {len(u)}")
    return u


def batch_utts():
    # Unfinished, just use single_utts.
    batch_size = 8
    utt_len = 32
    input_ids = []
    attention_masks = []
    # Tokenize utterances
    for utt in utts:
        encoded_dict = tokenizer(utt, add_special_tokens=True, padding='max_length', return_tensors='pt', truncation=True, return_attention_mask=True)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        # TODO: add utt idx to batch
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    # Use basic TensorDataset
    dataset = TensorDataset(input_ids, attention_masks)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Generate embeddings
    for batch in dataloader:
        b_input_ids = batch[0].to(device)
        b_attention_masks = batch[1].to(device)

        output = model(b_input_ids, attention_mask=b_attention_masks, token_type_ids=None)
        sent_emb = output[1]
        print(sent_emb)
        break


def single_utts(utts):
    # Generate utterance embeddings one-by-one

    i = 0
    for utt in utts:
        print(f"{i} / {len(utts)}") if not i % 100 else None
        encoded = tokenizer(utt)
        input_ids = torch.tensor(encoded['input_ids']).unsqueeze(0)
        att_mask = torch.tensor(encoded['attention_mask']).unsqueeze(0)
        output = model(input_ids, attention_mask=att_mask, token_type_ids=None)
        sent2emb[utt] = output[1]  # TODO: detach before saving
        i += 1

    pickle.dump(sent2emb, open("../Pickles/sent2emb.pickle", 'wb'))


utts = load_utts()
single_utts(utts)
