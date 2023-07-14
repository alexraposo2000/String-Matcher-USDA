from transformers import BertTokenizer, BertModel
import torch
import pandas as pd
import numpy as np
from clean import * # import the cleaner function from clean.py

def bert_match(list1, list2, type, join_punct = ', '):
    list1 = cleaner(list1,type,join_punct)
    list2 = cleaner(list2,type,join_punct)

    if type == 'lists':
        list1 = [', '.join(l1) for l1 in list1]
        list2 = [', '.join(l2) for l2 in list2]
    # Load pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    cos = []
    doc = []
    df_matches_data = []
    df_scores_data = []
    for query1 in list1:
        scores = []
        for query2 in list2:
            # Tokenize the sentences
            tokens = tokenizer.encode_plus(query1, query2, padding=True, truncation=True, return_tensors='pt')
            print(tokens)
            # Retrieve token IDs and attention mask
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']

            # Forward pass through BERT model
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

            # Extract the embeddings
            sentence1_embedding = outputs.last_hidden_state[0][0]  # Embedding for the first sentence
            sentence2_embedding = outputs.last_hidden_state[0][1]  # Embedding for the second sentence

            # Calculate cosine similarity between sentence embeddings
            cosine_similarity = torch.nn.functional.cosine_similarity(sentence1_embedding, sentence2_embedding, dim=0)
            cs = cosine_similarity.item()
            print('score',cs)
            scores.append(cs)

        # scores = np.array(scores)
        print('SCORES', scores)
        scores_copy = scores.copy()
        # max_value = max(scores_copy)
        list2_copy = list2.copy()
        top_3_matches = []
        top_3_scores = []
        for i in range(3):
            if len(scores_copy)>0:
                max_idx = scores_copy.index(max(scores_copy))
                top_3_matches.append(list2_copy[max_idx])
                top_3_scores.append(max(scores_copy))
                print(scores_copy[max_idx],list2_copy[max_idx])
                del scores_copy[max_idx]
                del list2_copy[max_idx]
            else:
                print('NONE')
                top_3_matches.append(None)
                top_3_scores.append(None)
        df_matches_data.append(top_3_matches)
        print(top_3_matches)
        df_scores_data.append(top_3_scores)

    df_matches = pd.DataFrame()
    df_matches['list1'] = list1
    df_matches['1'] = [d[0] for d in df_matches_data]
    df_matches['2'] = [d[1] for d in df_matches_data]
    df_matches['3'] = [d[2] for d in df_matches_data]
    df_scores = pd.DataFrame()
    df_scores['1'] = [d[0] for d in df_scores_data]
    df_scores['2'] = [d[1] for d in df_scores_data]
    df_scores['3'] = [d[2] for d in df_scores_data]
    return df_matches, df_scores

list_of_strings1 = ['turkey, cheese, cheddar', 'watermelon, lemon']
list_of_strings2 = ['turkey, cheese, cheddar', 'ham, lemon, mayo']
print(bert_match(list_of_strings1,list_of_strings2,type = 'strings'))
