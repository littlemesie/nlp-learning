# coding: utf-8

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

sentences = ["i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]

word_sequence = " ".join(sentences).split()
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word_dict = {w: i for i, w in enumerate(word_list)}
print(word_dict)

# word_to_ix = {"hello": 0, "world": 1}
# 2表示有2个词，5表示5维度，其实也就是一个2x5的矩阵
# embeds = nn.Embedding(2, 5)
# lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
# hello_embed = embeds(autograd.Variable(lookup_tensor))
# print(hello_embed)

embeds = nn.Embedding(len(word_list), 5)

sentence1 = torch.Tensor(3, 5)
sentence2 = torch.Tensor(3, 5)
for i, word in enumerate(sentences[0].split(" ")):

    lookup_tensor = torch.LongTensor([word_dict[word]])
    embed = torch.FloatTensor(embeds(autograd.Variable(lookup_tensor)))
    sentence1[i] = embed[0]
    # print(embed)

for i, word in enumerate(sentences[1].split(" ")):

    lookup_tensor = torch.LongTensor([word_dict[word]])
    embed = torch.FloatTensor(embeds(autograd.Variable(lookup_tensor)))
    sentence2[i] = embed[0]
    # print(embed)

sim = torch.cosine_similarity(sentence1, sentence2, dim=1)
print(sentence1)
print(sentence2)

print(sim)