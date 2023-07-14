from clean import *
from BERT_match import *
from fuzzy_match import *
from tf_idf_match import *

def do_map(l1,l2,type = "strings",preference = ""):
    """
    Type: indicates the objects to be mapped. "strings" (matchings strings in list l1 to strings in l2),
    "lists" (matching lists of strings in l1 to lists of strings in l2)
    Preference: "" indicates no preference in matching method and the ideal method will be selected for you,
    otherwise state a preferance ("BERT", "fuzzy", or "tf-idf")
    *Requirements: transformers, torch, fuzzywuzzy, sklearn, wordnet
    """
    if preference == "":
        fuzzy_df, fuzzy_scores  = fuzzy_match(l1,l1, type = type)
        tf_df, tf_scores = tf_idf_match(l1,l2,type = type)



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

print(do_map(list_of_strings1,list_of_strings2,type = 'strings',preference = 'BERT'))


# print(do_map(list_of_lists1,list_of_lists2,type = 'lists',preference = 'tf-idf'))
