from clean import *
from BERT_match import *
from fuzzy_match import *
from tf_idf_match import *
import numpy as np
import pandas as pd

def do_map(l1,l2,type = "strings",preference = ""):
    """
    Type: indicates the objects to be mapped. "strings" (matchings strings in list l1 to strings in l2),
    "lists" (matching lists of strings in l1 to lists of strings in l2)
    Preference: "" indicates no preference in matching method and the ideal method will be selected for you,
    otherwise state a preferance ("BERT", "fuzzy", or "tf-idf")
    *Requirements: transformers, torch, fuzzywuzzy, sklearn, wordnet
    """
    def pick_best(df1,df2,df3):
        final_matches = []
        df1_vs_df2 = (df1==df2).astype(int)
        df2_vs_df3 = (df2==df3).astype(int)
        df3_vs_df1 = (df3==df1).astype(int)
        total = df1_vs_df2+df2_vs_df3+df3_vs_df1
        for i in range(len(total)):
            if total.iloc[i,1] == 3: # if they all agree on top match
                final_matches.append(df1['1'][i])
            elif total.iloc[i,1] == 1: # if 2 methods agree on top match
                if df1_vs_df2.iloc[i,1] == 1:
                    final_matches.append(df1['1'][i])
                elif df2_vs_df3.iloc[i,1] == 1:
                    final_matches.append(df2['1'][i])
                elif df3_vs_df1.iloc[i,1] == 1:
                    final_matches.append(df3['1'][i])
            else: # if no methods agree on the top match take vote from top 3 instead
                list_1 = [df1.iloc[i,1],df1.iloc[i,2],df1.iloc[i,3]]
                list_2 = [df2.iloc[i,1],df2.iloc[i,2],df2.iloc[i,3]]
                list_3 = [df3.iloc[i,1],df3.iloc[i,2],df3.iloc[i,3]]
                ls = list_1+list_2+list_3
                unique = np.unique(ls)
                count_ls = []
                for u in unique:
                    count_ls.append(ls.count(u))
                # if there is a tie, pick the one with the higher ranking in any one of the methods
                # if it's still a tie, then pick randomly
                max_count = max(count_ls)
                if ls.count(max_count) >1:
                    tie = 0
                    idxs = [index for (index, item) in enumerate(count_ls) if item == max_count]
                    top_words = [unique[i] for i in idxs]
                    score = [0 for word in top_words]
                    while tie < 3:
                        for w in range(len(top_words)):
                            if top_words[w] == list_1[tie]:
                                score[w] += 1
                            if top_words[w] == list_2[tie]:
                                score[w] += 1
                            if top_words[w] == list_3[tie]:
                                score[w] += 1
                        if score.count(max(score)) == 1: # tie has been broken
                            final_matches.append(top_words[score.index(max(score))])
                            tie = 5
                        else:
                            tie += 1
                            if tie == 3: # tie hasn't been broken so pick randomly between top scoring words
                                tw_idx = [index for (index, item) in enumerate(score) if item == max(score)]
                                final_matches.append(np.random.choice(np.array([top_words[i] for i in tw_idx])))

                else: # no tie to break
                    final_matches.append(unique[unique.index(max_count)])
        return final_matches









    if preference == "": # perform all methods and do a concensus vote
        fuzzy_df, fuzzy_scores  = fuzzy_match(l1,l2, type = type)
        tf_df, tf_scores = tf_idf_match(l1,l2,type = type)
        BERT_df, BERT_scores = bert_match(l1,l2, type = type)
        matches = pick_best(fuzzy_df,tf_df,BERT_df)
        df = pd.DataFrame()
        df['list1'] = tf_df['list1']
        df['list2 matches'] = matches
        return df

    else:
        ls = [bert_match,fuzzy_match,tf_idf_match]
        dictionary = {"BERT":0,"fuzzy":1, "tf-idf":2}
        return ls[dictionary[preference]](l1,l2, type = type)


list_of_lists1 = [['bananas and','water','strawberries'],['whole wheat','yeast','water'],['corn flour','butter']]
list_of_lists2 = [['bananas and','strawberries'],['whole wheat','salt','yeast','water'],['corn flour','salted butter','yeast']]
# print(do_map(list_of_lists1,list_of_lists2,type = 'lists',preferance = 'BERT'))

list_of_strings1 = [
    "I like to eat apples.",
    "Apples are delicious fruits.",
    "Bananas are yellow in color."
]
list_of_strings2 = [
    "I like to eat.",
    "Apples are fruits.",
    "yellow is a color."
]

print(do_map(list_of_strings1,list_of_strings2,type = 'strings',preference = ''))


# print(do_map(list_of_lists1,list_of_lists2,type = 'lists',preference = 'tf-idf'))
