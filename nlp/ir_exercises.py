'''Exercises from the book Introduction to Information Retrieval: https://nlp.stanford.edu/IR-book/'''

import pandas as pd

# ### Ex 1.1: Build inverted index
# docs = {'doc1': 'new home sales top forecasts',
# 'doc2': 'home sales rise in july',
# 'doc3': 'increase in home sales in july',
# 'doc4': 'july new home sales rise'}
#
# res = {}
# for key, value in docs.items():
#     words = value.split(' ')
#     for word in words:
#         if word in res.keys():
#             res[word].append(key)
#         else:
#             res[word] = [key]
#
# print(res)
# # OK

### Ex 1.2: Build term-document incidence matrix
docs = {'doc1': 'breakthrough drug for schizophrenia',
'doc2': 'new schizophrenia drug',
'doc3': 'new approach for treatment of schizophrenia',
'doc4': 'new hopes for schizophrenia patients'}

#build indices: build list of distict words in corpus
indices = []
for text in docs.values():
    # new_words = [word for word in text.split(' ') if word not in indices]
    # indices += new_words
    # alternative: with set difference
    indices += list(set(text.split(' ')) - set(indices))

indices.sort()

res = {'word': indices}
for key, value in docs.items():
    res[key] = [int(word in value.split(' ')) for word in indices]

res = pd.DataFrame(res).set_index('word')
print(res)
# OK






