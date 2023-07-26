# Concensus voting string matching using fuzzy string matching, tf-idf, and BERT
This project was developed in collaboration with the Lemay lab, part of the USDA immunity and disease prevention unit, for mapping betten related databases using string matching. All scripts are written in Python and were run using Python version 3.10.
This method and handle maps between lists of strings and lists containing lists of strings.
<ins>**Mapping between lists of strings:**</ins> after isolating a starting list and a target list from two datasets, modify mapper.py in the designated area at the bottom of the file to access these lists.</br>
Example input: list1 = ['banana','apple','butter','corn bread'], list2 = ['apple sauce','banana bread,'cheddar cheese','corn tortilla']</br>

<ins>**Mapping between lists of lists containings strings:**</ins> after isolating a starting list containing lists of strings and a target list containing lists of strings from two datasets, modify mapper.py in the designated area at the bottom of the file to access these lists. 
The importance of this second case is specific to the mapper's use in the Lemay lab to handle mappings between lists of ingredients.</br>
Example input: list1 = [['banana','apple'],['butter', 'corn bread']], list2 = [['apple sauce','banana bread],['cheddar cheese','corn tortilla']]</br>

NOTE: multiple terms in the starting list can be mapped to a single term in the target list (for this reason, mapping l1 to l2 will not always result in the same mapping from l2 to l1).
## File descriptions
**clean.py** - this file preprocesses strings 
## Running the mapper on your own data
Demo data provided
