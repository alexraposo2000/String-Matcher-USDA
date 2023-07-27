from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from clean import *
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from clean import *

# stopwords = ['','and', 'to', 'not', 'no',  'bkdfrd', 'ppd', 'pkgddeli', 'pkgd', 'xtra', 'oz', 'in', 'with', 'or', 'only', 'cooking', 'as', 'food', 'distribution', 'form', 'w', 'wo', 'ns', 'nfs', 'incl']
stopwords = StopWords()
def preprocess(text,type = 'strings'):
    if type == 'lists': # a list of lists containing strings
        out = []
        for l in text:
            # print('item in text',l)
            out.append(preprocess(l, type = 'strings'))
        words = out
    else:
    # Tokenize the text into words
        # print('string call',text)
        words = nltk.word_tokenize(text.lower())
        words = [word for word in words if word not in stopwords]
        # print('WORDS',words)
    # Lemmatize words to their base form
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]
        words = [word for word in words if word not in stopwords]

    # Join the words back into a string
    # print(' '.join(words))
    return ' '.join(words)

def tf_idf_string_matching(strings, query,type):
    # Preprocess the input strings and the query
    preprocessed_strings = [preprocess(s,type) for s in strings]
    # print('preprocessed_strings',preprocessed_strings)
    # if type == 'lists':
    #     preprocessed_query = ' ,'.join([''.join(j) for j in cleaner(query,type)])
    # else:
    preprocessed_query = preprocess(query,type)

    # print('TYPE',preprocessed_query.type())
    # print('preprocessed_query',preprocessed_query)
    # Combine the input strings and query into a single list for TF-IDF vectorization
    documents = preprocessed_strings + [preprocessed_query]

    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    # Compute TF-IDF matrix
    tfidf_matrix = vectorizer.fit_transform(documents)
    # print('MADE IT')
    # Calculate cosine similarity between query vector and document vectors
    query_vector = tfidf_matrix[-1]  # Last row corresponds to the query
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix[:-1])  # Exclude the query from the matrix

    # Get the indices of the most similar strings
    most_similar_indices = cosine_similarities.argsort()[0][::-1]  # Sort in descending order

    # Get the most similar strings and their cosine similarity scores
    best_matches = [(strings[idx], cosine_similarities[0, idx]) for idx in most_similar_indices]

    return best_matches

# TESTING

# print(ls2[0:5])
# strings = ["I like to eat apples.", "Apples are delicious fruits.", "Bananas are yellow in color."]
# queries = ["I like eating apples."]

# matches = tfidf_string_matching(strings, query)
# print("Best Matches:")
def tf_idf_match(queries, strings, type,clean_punct = ', '):
    match_terms = []
    match_scores = []
    for query in queries:
        # print('QUERY',query)
        terms = []
        scores = []
        matches = tf_idf_string_matching(strings, query,type)
        for match, similarity in matches:
            terms.append(match)
            scores.append(similarity)
        match_terms.append(terms)
        match_scores.append(scores)

    final_scores = []
    final_matches = []
    for i in range(len(match_scores)):
        top_3_terms = []
        top_3_scores = []
        ith_scores = match_scores[i].copy()
        ith_list = match_terms[i].copy()
        for j in range(3):
            if len(ith_scores)>0:
                idx = ith_scores.index(max(ith_scores))
                # match_scores[i][idx]
                top_3_terms.append(ith_list[idx])
                top_3_scores.append(ith_scores[idx])
                del ith_list[idx]
                del ith_scores[idx]
            else:
                top_3_terms.append(None)
                top_3_scores.append(None)
        final_matches.append(top_3_terms)
        final_scores.append(top_3_scores)
    df = pd.DataFrame()
    # if type == 'lists':
    #     queries = cleaner(queries, type, clean_punct)
    #     queries = [', '.join(q) for q in queries]
    #     print('QUERIES!!',queries)
    #     print(final_matches)
    #     final_matches = [cleaner(f,type, clean_punct) for f in final_matches]
    #     final_matches = [', '.join(m) for m in final_matches]
    if type == 'lists':
        queries = cleaner(queries, type, clean_punct)
        queries = [', '.join(l1) for l1 in queries]
        df['list1'] = queries
        one = cleaner([f[0] for f in final_matches],type,clean_punct)
        one = [', '.join(l1) for l1 in one]
        df['1'] = one
        two = cleaner([f[1] for f in final_matches],type,clean_punct)
        two = [', '.join(l1) for l1 in two]
        df['2'] = two
        three = cleaner([f[2] for f in final_matches],type,clean_punct)
        three = [', '.join(l1) for l1 in three]
        df['3'] = three

        df2 = pd.DataFrame()
        df2['1'] = [f[0] for f in final_scores]
        df2['2'] = [f[1] for f in final_scores]
        df2['3'] = [f[2] for f in final_scores]
        return df, df2
    else:
        df['list1'] = queries
        df['1'] = [f[0] for f in final_matches]
        df['2'] = [f[1] for f in final_matches]
        df['3'] = [f[2] for f in final_matches]

        df2 = pd.DataFrame()
        df2['1'] = [f[0] for f in final_scores]
        df2['2'] = [f[1] for f in final_scores]
        df2['3'] = [f[2] for f in final_scores]
        return df, df2

# queries = pd.read_csv('./demo_data/data.csv')
# queries = list(queries.iloc[:,0])
# # print(ls1[0:5])
# strings = pd.read_csv('./demo_data/data2_subset.csv')
# strings = list(strings.iloc[:,0])
# df_matches,df_scores = tf_idf_match(strings, queries, type = 'strings')


# queries = [['bananas and','water','strawberries'],['whole wheat','yeast','water'],['corn flour','butter']]
# strings = [['bananas and','strawberries'],['whole wheat','salt','yeast','water'],['corn flour','salted butter','yeast']]
# df_matches,df_scores = tf_idf_match(strings, queries, type = 'lists')

# print(df_matches[['list1','1']])
# print(df_matches['2'])
