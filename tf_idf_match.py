from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import nltk
import pandas as pd
import numpy as np
# nltk.download('wordnet')
# wn = nltk.WordNetLemmatizer()
from clean import * # import the cleaner function from clean.py

def tf_idf_match(list1, list2, type, join_punct = ', ',top_3 = False):
    list1 = cleaner(list1,type,join_punct)
    list2 = cleaner(list2,type,join_punct)
    if type == 'lists':
        list1 = [', '.join(l1) for l1 in list1]
        list2 = [', '.join(l2) for l2 in list2]
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    # Compute TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(list2)
    df_matches_data = []
    df_scores_data = []
    for query in list1:
        top_3_scores = []
        top_3_matches = []
        # Transform the query string into a TF-IDF vector
        query_vector = vectorizer.transform([query])
        # Calculate cosine similarity between query vector and document vectors
        cosine_similarities = list(cosine_similarity(query_vector, tfidf_matrix).flatten())
        list2_copy = list2.copy()
        # print('COS SIMS',cosine_similarities)
        for i in range(3):
            if len(cosine_similarities)>0:
                most_similar_document_index = np.array(cosine_similarities).argmax()
                top_3_scores.append(max(cosine_similarities))
                # Get the most similar document
                most_similar_document = list2_copy[most_similar_document_index]
                top_3_matches.append(most_similar_document)
                del cosine_similarities[most_similar_document_index]
                del list2_copy[most_similar_document_index]
            else:
                top_3_matches.append(None)
                top_3_scores.append(None)
        df_matches_data.append(top_3_matches)
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

# TESTING
list_of_lists1 = [['bananas and','water','strawberries'],['whole wheat','yeast','water'],['corn flour','butter']]
list_of_lists2 = [['bananas and','strawberries'],['whole wheat','salt','yeast','water'],['corn flour','salted butter','yeast']]
# print(tf_idf_match(list_of_lists1,list_of_lists2,type = 'lists'))

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
# print(tf_idf_match(list_of_strings1,list_of_strings2,type = 'strings', join_punct =' '))
print(tf_idf_match(list_of_lists1,list_of_lists2,type = 'lists', join_punct =' '))

# print(f"Query: {query}")
# print(f"Most similar document: {most_similar_document}")
