from clean import *
from BERT_match import *
from fuzzy_match import *
from tf_idf_match import *
import numpy as np
import pandas as pd
import random

def vote(df1,df2,df3):
    def tie_break(i):
        ls_1_2 = [df1.iloc[i,1],df2.iloc[i,1],df3.iloc[i,1],df1.iloc[i,2],df2.iloc[i,2],df3.iloc[i,2]]
        ls_1_2 = [l for l in ls_1_2 if (l in [df1.iloc[i,1],df2.iloc[i,1],df3.iloc[i,1]])] # make sure we're only counting votes for our original top 3
        unique_ls_2 = list(pd.Series(ls_1_2).unique())
        votes_2 = []
        for u in unique_ls_2:
            votes_2.append(ls_1_2.count(u))
        find_tie = [i for i in range(len(votes_2)) if votes_2[i] == max(votes_2)]

        if len(find_tie)>1:# NEED TO CHECK FOR A TIE HERE!
            ls_1_3 = [df1.iloc[i,1],df2.iloc[i,1],df3.iloc[i,1],df1.iloc[i,2],df2.iloc[i,2],df3.iloc[i,2],df1.iloc[i,3],df2.iloc[i,3],df3.iloc[i,3]]
            ls_1_3 = [l for l in ls_1_3 if (l in [df1.iloc[i,1],df2.iloc[i,1],df3.iloc[i,1]])] # make sure we're only counting votes for our original top 3
            unique_ls_3 = list(pd.Series(ls_1_3).unique())
            votes_3 = []
            for u in unique_ls_3:
                votes_3.append(ls_1_3.count(u))
            find_tie_3 = [i for i in range(len(votes_3)) if votes_3[i] == max(votes_3)]
            if len(find_tie_3)>1:
                return random.choice([unique_ls_3[i] for i in find_tie_3])
            else:
                return unique_ls_3[find_tie_3[0]]
        else:
            return unique_ls_2[votes_2.index(max(votes_2))]

    matches = []
    for i in range(len(df1)):
        ls = [df1.iloc[i,1],df2.iloc[i,1],df3.iloc[i,1]]
        unique_ls = list(pd.Series(ls).unique())
        votes = []
        for u in unique_ls:
            votes.append(ls.count(u))
        if max(votes) == 1: # ie, there is a tie among the 3 methods for 1st place
            matches.append(tie_break(i))
        else:
            matches.append(unique_ls[votes.index(max(votes))])
    matches_df = pd.DataFrame()
    matches_df['list1'] = df1.iloc[:,0]
    matches_df['matches'] = matches
    return matches_df

def do_map(l1,l2,type,preference = ""):
    if preference == "": # perform all methods and do a concensus vote
        fuzzy_df, fuzzy_scores  = fuzzy_match(l1,l2, type = type)
        print("fuzzy match complete")
        # print('fuzzy',fuzzy_df)
        tf_df, tf_scores = tf_idf_match(l1,l2,type = type)
        print("tf-idf match complete")
        # print('tf-idf',tf_df)
        BERT_df, BERT_scores = bert_match(l1,l2, type = type)
        print("BERT match complete")
        # print('bert',BERT_df)
        print("voting has begun")
        matches = vote(fuzzy_df,tf_df,BERT_df)
        print("voting complete")
        return matches

    else:
        ls = [bert_match,fuzzy_match,tf_idf_match]
        dictionary = {"BERT":0,"fuzzy":1, "tf-idf":2}
        return ls[dictionary[preference]](l1,l2, type = type)

# -----------------------------------------------------------------------------
# MODIFY THIS SECTION TO ADD YOUR OWN DATA!
# -----------------------------------------------------------------------------
# read in list1
list1 = list(pd.read_csv('./demo_data/data.csv')['Response'])
# print('list1',list1)
# read in list2
list2 = list(pd.read_csv('./demo_data/data2_subset.csv')['name'])
# print('list2',list2)


# TO MAP BETWEEN 2 LISTS OF LISTS CONTAINING STRINGS (instead of lists of strings), change type = "strings" to type = "lists"
# Uncomment the next 4 lines to map between lists of strings with no preferred method:
# ------------------------------------------------------------------------------------
# list2 = ["I like to eat apples.", "Apples are delicious fruits.", "Bananas are yellow in color."]
# list1 = ["I like eating apples.", "fruits are delicious", "yellow is my favorite color"]
# matches = do_map(list1,list2,type = 'strings',preference = "")
# print(matches) # print result to your terminal
# matches.to_csv('matches.csv') # save the results as a csv in your working directory

# Uncomment the next 4 lines to map between lists of strings with a method:
# ------------------------------------------------------------------------------------
# matches,scores = do_map(list1,list2,type = 'strings',preference = "BERT") # specify a preference if wanted ("fuzzy", "tf-idf", or "BERT")
# print(matches) # print result to your terminal
# matches.to_csv('matches.csv') # save mapping
# scores.to_csv('scores.csv') # save scores
# -----------------------------------------------------------------------------
# END SECTION TO MODIFY
# -----------------------------------------------------------------------------
