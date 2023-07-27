from fuzzywuzzy import fuzz
import re
import string
# import nltk
import pandas as pd
# nltk.download('wordnet')
# wn = nltk.WordNetLemmatizer()
from clean import * # import the cleaner function from clean.py


def fuzzy_match(list1, list2, type, clean = True, clean_punct = ', '):
    '''clean (defaults to True) will clean the contents of list1 and list2 if set to True
    lst1 is a list of lists or a list of strings to match to objects in lst2
    type = 'lists' or = 'strings' indicates whether you're matching lists to lists or strings to strings'''
    # clean the lists (see cleaner documentation above)
    list1 = cleaner(list1, type, clean_punct)
    list2 = cleaner(list2, type, clean_punct)
    # if we have lists of lists, make each list into a string.
    if type == 'lists':
        list1 = [', '.join(l1) for l1 in list1]
        list2 = [', '.join(l2) for l2 in list2]
    # Calculate the similarity scores using fuzz.ratio() for each pair of strings
    # We will determine the top 3 matches *if there are more than 3 items in list2*
    df_matches = pd.DataFrame()
    df_matches_data = []
    df_scores = pd.DataFrame()
    df_scores_data = []
    for l1 in list1:
        scores = []
        for l2 in list2:
            list2_copy = list2.copy()
            score = fuzz.ratio(l1, l2)
            scores.append(score)
        # return top 3 matches
        top_3_matches = []
        top_3_scores = []
        for i in range(3):
            if len(scores)>0:
                max_idx = scores.index(max(scores))
                top_3_matches.append(list2_copy[max_idx])
                top_3_scores.append(max(scores))
                del scores[max_idx]
                del list2_copy[max_idx]
            else:
                top_3_matches.append(None)
                top_3_scores.append(None)
        df_matches_data.append(top_3_matches)
        df_scores_data.append(top_3_scores)

    df_matches['list1'] = list1
    df_matches['1'] = [d[0] for d in df_matches_data]
    df_matches['2'] = [d[1] for d in df_matches_data]
    df_matches['3'] = [d[2] for d in df_matches_data]

    df_scores['1'] = [d[0] for d in df_scores_data]
    df_scores['2'] = [d[1] for d in df_scores_data]
    df_scores['3'] = [d[2] for d in df_scores_data]


    # print(df_matches)
# df = pd.DataFrame()
# df['list1'] = list1
# df['list2 matches'] = doc
# df['scores'] = cos

    print('MATCHES',df_matches)
    return df_matches, df_scores

# queries = pd.read_csv('./demo_data/data.csv')
# queries = list(queries.iloc[:,0])
# # print(ls1[0:5])
# strings = pd.read_csv('./demo_data/data2_subset.csv')
# strings = list(strings.iloc[:,0])
# #
# d1, d2 = fuzzy_match(queries, strings, type = 'strings', clean = True, clean_punct = ', ')
# # print(d1[['list1','1']])
# # string1 = ['apple','sugar and','water']
# # # string1 = ', '.join(string1)
# # string2 = ['sugar','apple']
# # # string2 = ', '.join(string2)
# # string3 = ['banana','sugar']
# # string3 = ', '.join(string3)
#
# # Test out cleaning method
# # print(cleaner([string1,string2,string3],type = 'lists'))
# # print(cleaner(string1,type = 'strings'))
# # print(cleaner(', '.join(string1),type = 'string',join_punct = '!'))
#
# # Test the matching
# # d1, d2 = fuzzy_match(string1, string2, type = 'strings', clean = True, clean_punct = ', ')
# # d1, d2 = fuzzy_match([string1, string2],[string2,string3], type = 'lists', clean = True, clean_punct = ', ')
# #
# # print('D1:',d1)
# #
# # print('D2:',d2)
#
#
#
#
#
# # Import packages
# # import pandas as pd
# # import string
# # import re
# # from polyfuzz.models import TFIDF
# # from polyfuzz import PolyFuzz
# # import nltk
# # wn = nltk.WordNetLemmatizer()
# #
# # #Load data
# # discont = pd.read_csv('../data/03/wweia_discontinued_foodcodes.csv')
# # current = pd.read_csv('../data/02/fndds_16_18_all.csv')
# #
# # punct = string.punctuation[0:11] + string.punctuation[13:] # remove '-' from the list of punctuation. This is needed for the text cleaner in the following cell
# #
# # stopwords = ['','and', 'to', 'not', 'no',  'bkdfrd', 'ppd', 'pkgddeli', 'pkgd', 'xtra', 'oz', 'in', 'with', 'or', 'only', 'cooking', 'as', 'food', 'distribution', 'form', 'w', 'wo', 'ns', 'nfs', 'incl']
# #
# # def clean_text(text):
# #     text = "".join([word for word in text if word not in punct])
# #     tokens = re.split('[-\W+]', text)
# #     text = [word for word in tokens if word not in stopwords]
# #     text = [wn.lemmatize(word) for word in tokens if word not in stopwords]
# #     return ' '.join(text)
# #
# # discont['DRXFCLD_clean'] = discont['DRXFCLD'].apply(lambda x: clean_text(x.lower()))
# # current['parent_desc_clean'] = current['parent_desc'].apply(lambda x: clean_text(x.lower()))
# #
# # discont_list = discont['DRXFCLD_clean'].to_list()
# # current_list = current['parent_desc_clean'].to_list()
# #
# # tfidf = TFIDF(n_gram_range=(3, 3))
# # model = PolyFuzz(tfidf).match(discont_list, current_list)
# #
# # match_str = model.get_matches()
# # match_str.rename(columns={'From':'DRXFCLD_clean', 'To':'parent_desc_clean'},inplace=True)
# # fndds_matched = match_str.merge(discont, on='DRXFCLD_clean', how='left')
# # fndds_matched_ = fndds_matched.merge(current, on='parent_desc_clean', how='left').drop_duplicates(subset='DRXFDCD')
# #
# # fndds_matched_[['DRXFDCD', 'DRXFCLD', 'parent_foodcode', 'parent_desc', 'Similarity']].sort_values('Similarity', ascending=False).to_csv('../data/03/string_match.csv', index=None)
