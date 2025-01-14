import importlib
from tqdm import tqdm
import pandas as pd
import spacy
import pickle
from collections import defaultdict, Counter
import numpy as np
import torch

import fasttext.util


language_source = "zh_core_web_sm"
csv_dir = [f"data/csldaily/data.tsv"]

def get_parts_of_speech(sentence):
    doc = nlp(sentence)
    normalized_sentence = [token.lemma_ for token in doc]
    pos_tags = [(token.text, token.pos_) for token in doc]
    return [(n.lower(), p[0],p[1]) for n, p in zip(normalized_sentence, pos_tags)]


dict_pos = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary verb",
    "CONJ": "conjunction",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "NUM": "numeral",
    "PART": "particle",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "X": "other",
}

selected_vocab = ["NOUN", "NUM", "ADV", "PRON", "PROPN", "ADJ", "VERB"]

nlp = spacy.load(language_source)

dfs = []
for csv in csv_dir:
    df = pd.read_csv(csv, sep='\t')
    dfs.append(df)
df = pd.concat(dfs)

df.label_char = df.label_char.apply(eval)
df["translation"] = df.label_char.str.join('')
df.label_gloss = df.label_gloss.apply(eval)
df.label_word = df.label_word.apply(eval)


label_glosses = [item for sublist in list([x for x in df.label_word.values]) for item in sublist]
translations = df.translation.values

dict_sentence = {}
all_lens = []
for sentence in tqdm(translations):
    pos = get_parts_of_speech(sentence)
    lems = []
    for lem, word, part in pos:
        
        if part in selected_vocab:
            lems.append(word)
            
    all_lens.extend(lems)
    dict_sentence[sentence] = lems

fasttext.util.download_model('zh', if_exists='ignore') 
ft = fasttext.load_model('cc.zh.300.bin')

dict_lem_to_id = {l:i for i, l in enumerate(list(set(all_lens)))}

# flat_list = [item for sublist in list(dict_sentence.values()) for item in sublist]
# count = Counter(flat_list)

# vector = torch.zeros((len(dict_lem_to_id),300))
# for key, value in tqdm(dict_lem_to_id.items()):
#     vector[value] = torch.tensor(ft.get_word_vector(key))

# dict_id_to_lem = {v:k for k, v in dict_lem_to_id.items()}

dict_processed_words = {
    "dict_lem_counter": dict(Counter(all_lens)),
    "dict_sentence": dict(dict_sentence),
    "dict_lem_to_id": dict_lem_to_id
}

with open("data/csldaily/processed_words.csl_pkl", "wb") as f:
    pickle.dump(dict_processed_words, f)