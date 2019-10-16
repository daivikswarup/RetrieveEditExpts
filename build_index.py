import nltk
import os
import numpy as np
import csv
import re
from sklearn.feature_extraction.text import TfidfVectorizer

dire = 'wlpdata'

files = list(set([x.split('.')[0] for x in os.listdir(dire) if
                  re.match('protocol_[\d]*.(ann|txt)', x) ]))

def process_file(protocol, entities):
    lastindex = 0
    newstring = ''
    with open(entities, 'r') as f:
        reader = csv.reader(f, delimiter = '\t')
        for line in reader:
           if line[0].startswith('T'):
                tag = line[1].split()[0]
                occurences = line[1][len(tag):].split(';')
                tag = '~' + tag + '~'
                for occurence in occurences:
                    split = occurence.split()
                    start = int(split[0])
                    end = int(split[1])
                    term = line[2]
                    newstring += protocol[lastindex:start] + ' ' + tag+' '
                    lastindex = end + 1
        newstring += protocol[lastindex:]
    return zip(newstring.splitlines(), protocol.splitlines())

sentences = []
for fil in files:
    with open(os.path.join(dire, '%s.txt'%fil), 'r') as f:
        protocol = f.read()
    sentences.extend(process_file(protocol, os.path.join(dire, '%s.ann'%fil)))

replaced, originals = zip(*sentences)
with open('replaced_sentences.txt', 'w') as f:
    f.write('\n'.join(replaced))

with open('original_sentences.txt', 'w') as f:
    f.write('\n'.join(originals))

# Get TF IDF representation for each sentence
vectorizer = TfidfVectorizer(ngram_range = (1, 3))

# Representation of original sentences
X_orig = vectorizer.fit_transform(originals)

# Representations of replaced sentences
X_rep = vectorizer.fit_transform(replaced)
