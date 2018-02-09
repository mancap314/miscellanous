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

# ### Ex 1.2: Build term-document incidence matrix
# docs = {'doc1': 'breakthrough drug for schizophrenia',
# 'doc2': 'new schizophrenia drug',
# 'doc3': 'new approach for treatment of schizophrenia',
# 'doc4': 'new hopes for schizophrenia patients'}
#
# #build indices: build list of distict words in corpus
# indices = []
# for text in docs.values():
#     # new_words = [word for word in text.split(' ') if word not in indices]
#     # indices += new_words
#     # alternative: with set difference
#     indices += list(set(text.split(' ')) - set(indices))
#
# indices.sort()

# res = {'word': indices}
# for key, value in docs.items():
#     res[key] = [int(word in value.split(' ')) for word in indices]
#
# res = pd.DataFrame(res).set_index('word')
# print(res)
# # OK

# ### AND merge algorithm (intersection)
# def and_merge(l1, l2):
#     result = []
#     i1, i2 = 0, 0
#     while i1 < (len(l1) - 2) or i2 < (len(l2) - 2):
#         print('i1 = {}, i2 = {}'.format(i1, i2))
#         if l1[i1] == l2[i2]:
#             result.append(l1[i1])
#             i1 += 1
#             i2 += 1
#             continue
#         if l1[i1] < l2[i2]:
#             i1 += 1
#         else:
#             i2 += 1
#     return result
#
# l1 = [1, 3, 6, 9, 13, 18]
# l2 = [3, 13, 18, 24, 51]
# print(and_merge(l1, l2))

# Ex 1.11
def and_not_merge(l1, l2):
    '''delta between two lists'''
    result = []
    i1, i2 = 0, 0
    while i1 < (len(l1) - 2) or i2 < (len(l2) - 2):
        if l1[i1] < l2[i2]:
            result.append(l1[i1])
            i1 += 1
            continue
        if l2[i2] < l1[i1]:
            result.append(l2[i2])
            i2 += 1
            continue
        i1 += 1
        i2 += 1
    if i1 < len(l1) - 1:
        result += l1[i1:]
    if i2 < len(l2) - 1:
        result += l2[i2:]
    return result

l1 = [1, 3, 6, 9, 13, 18]
l2 = [3, 13, 18, 24, 51]
print(and_not_merge(l1, l2))

# On Google, burglar OR burglar: 34,6M results, AND: 25.6









