# Concensus voting string matching using fuzzy string matching, tf-idf, and BERT
This project was developed in collaboration with the Lemay lab, part of the USDA immunity and disease prevention unit, for mapping betten related databases using string matching. All scripts are written in Python and were run using Python version 3.10.
This method and handle maps between lists of strings and lists containing lists of strings.</br>

<ins>**Mapping between lists of strings:**</ins> after isolating a starting list and a target list from two datasets, modify mapper.py in the designated area at the bottom of the file to access these lists.</br>
Example input: list1 = ['banana','apple','butter','corn bread'], list2 = ['apple sauce','banana bread,'cheddar cheese','corn tortilla']</br>

<ins>**Mapping between lists of lists containings strings:**</ins> after isolating a starting list containing lists of strings and a target list containing lists of strings from two datasets, modify mapper.py in the designated area at the bottom of the file to access these lists. 
The importance of this second case is specific to the mapper's use in the Lemay lab to handle mappings between lists of ingredients.</br>
Example input: list1 = [['banana','apple'],['butter', 'corn bread']], list2 = [['apple sauce','banana bread],['cheddar cheese','corn tortilla']]</br>

NOTE: multiple terms in the starting list can be mapped to a single term in the target list (for this reason, mapping l1 to l2 will not always result in the same mapping from l2 to l1).
## File descriptions
**clean.py** - this file is called by fuzzy_match.py, tf_idf_match.py, and BERT_match.py. It preprocesses strings by removing stopwords, punctuation, and places the text in lower case. Stopwords can be customized my adding or removing strings from its stopwords list.</br>

**mapper.py** - when called on two lists (either lists of strings or lists of lists containing strings), a mapping from the first list (list1) provided and the second list provided (list2) is returned. If a desired method (fuzzy, tf-idf, or BERT) is specified, this method is used to perform the map and two dataframes will be returned. The first dataframe contains list1 and columns for each of its top 3 matches. The second dataframe contains the associated scores computed by that method (position (0,0) contains the similarity score of the specified method comparing the first item of list1 and the first item of list2 and so on). If no preferred method is specified, all 3 methods are run. The result of the 3 mappings are compared to determine a final mapping through a concensus vote. For each term in list1, the top match of each method is compared. If there is a 3-way-tie, the votes are re-tallied including votes from the 2nd place match from each method (only votes for the initial terms in the 3-way-tie are considered). If there is still a tie between 2 or more terms, votes for these tied terms are re-tallied including the 3rd place match from the methods. If a concensus is still not reached, a random selection is made from the tied terms with the most votes after this iterative process. To map between your own data, modify the file paths to read in your desired lists (see instructions in comments of the designated section at the bottom of mapper.py). This file references fuzzy_match.py, tf_idf_match.py, and BERT_match.py.
## Running the mapper on your own data
Demo data provided
