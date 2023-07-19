# This code is primarily taken from this article: https://towardsdatascience.com/bert-for-measuring-text-similarity-eec91c6bf9e1
from clean import *
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
def bert_match(list1, list2, type, join_punct = ', '):
    list1 = cleaner(list1,type,join_punct)
    list2 = cleaner(list2,type,join_punct)

    if type == 'lists':
        list1 = [', '.join(l1) for l1 in list1]
        list2 = [', '.join(l2) for l2 in list2]

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    sentences = list1+list2

    # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=128,
                                           truncation=True, padding='max_length',
                                           return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    # ------------------------------------------------------
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    from sklearn.metrics.pairwise import cosine_similarity
    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()
    # calculate similarities and return top 3
    df_scores_data = []
    df_matches_data = []
    for i in range(len(list1)):
        top_3_scores = []
        top_3_matches = []
        list2_sims = list(cosine_similarity([mean_pooled[i]],mean_pooled[len(list1):])[0])
        list2_copy = list2.copy()
        for j in range(3):
            if len(list2_sims)>0:
                max_idx = list2_sims.index(max(list2_sims))
                max_term = list2_copy[max_idx]
                top_3_matches.append(max_term)
                max_score = list2_sims[max_idx]
                top_3_scores.append(max_score)
                del list2_sims[max_idx]
                del list2_copy[max_idx]
            else:
                top_3_matches.append(None)
                top_3_scores.append(None)
        df_scores_data.append(top_3_scores)
        df_matches_data.append(top_3_matches)

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
