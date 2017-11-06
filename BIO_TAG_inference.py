# -*- coding: utf-8 -*-
#open('/Users/lmy/Dropbox/Personal/Coursework/CIS700-006/Project/POS_tagger_trained_on_Universal_Dependency_French_corpus/file.txt').read().decode('utf-8').split()

# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
import codecs 
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

import argparse
parser = argparse.ArgumentParser(description='Logistic regression BIO.')
parser.add_argument('--train', help='Training file')
parser.add_argument('--test', help='Test file')
parser.add_argument('--embed', help='Character embedding file')
args = parser.parse_args()

train_path = args.train
test_path = args.test

#train_path = '/Users/lmy/Dropbox/Personal/Coursework/CIS700-006/Project/UD_CH/zh-ud-train.conllu'
#train_path = '/Users/lmy/Dropbox/Personal/Coursework/CIS700-006/Project/BIO_TAG/file.txt'
#test_path = '/Users/lmy/Dropbox/Personal/Coursework/CIS700-006/Project/UD_CH/zh-ud-dev.conllu'
Pu='~`!@#$%^&*()_-+={[}]|\:;"\'<,>.?/'
Universal_tag_set = set()

def features(sentence, index, tags_previous_index = None):
    #""" sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prev_word': '<s>' if index == 0 else sentence[index - 1],       
        'C-2': '<s>' if index <= 1 else sentence[index - 1],              
        'next_word': '</s>' if index == len(sentence) - 1 else sentence[index + 1],       
        'C2': '</s>' if index >= len(sentence) - 2 else sentence[index + 2],
        'C1C2': '</s>' if index >= len(sentence) - 2 else sentence[index+1]+sentence[index+2],
        'C0C1': '</s>' if index == len(sentence) - 1 else sentence[index]+sentence[index+1],
        'C-1C0': '</s>' if index == 0     else sentence[index-1]+sentence[index],
        'C-2C-1': '</s>' if index <= 1    else sentence[index-2]+sentence[index-1],
        'C-1C1': '</s>' if (index == 0 or index ==  len(sentence) - 1 ) else sentence[index-1]+sentence[index+1],
        'Pu(C0)':   (sentence[index - 1] in Pu) ,
        'is_numeric': sentence[index].isdigit(),
        'T-1': '<s>' if index == 0 else (None if tags_previous_index is None else tags_previous_index[index-1])
    }
   
'''
def features(sentence, index):
    #""" sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prev_word': '<s>' if index == 0 else sentence[index - 1],
        'next_word': '</s>' if index == len(sentence) - 1 else sentence[index + 1],
        'is_numeric': sentence[index].isdigit(),
    }
'''



def untag(tagged_sentence):
    return [w for w, t in tagged_sentence] 
    
def gen_corpus(path):
    doc = []
    tagset = set()
    file = codecs.open(path, encoding='utf-8') 
    #with open(path, encoding='utf-8') as file:
    for line in file:
        if line[0].isdigit():
            features = line.split()
            word, pos= features[1], features[3]                
            if pos != "_":
                if(len(word)>1):
                    tagset.add('B'+pos)
                    if(not ('B'+pos) in Universal_tag_set):
                        Universal_tag_set.add('B'+pos)
                    tagset.add('I'+pos)
                    if(not ('I'+pos) in Universal_tag_set):
                        Universal_tag_set.add('I'+pos)
                    for order in range(len(word)):
                        if(order==0):
                            doc.append((word[order], 'B'+pos)) 
                        if(order!=0):
                            doc.append((word[order], 'I'+pos))
                else:
                    tagset.add('B'+pos)
                    if(not ('B'+pos) in Universal_tag_set):
                        Universal_tag_set.add('B'+pos)
                    doc.append((word, 'B'+pos))
                    
        elif len(line.strip()) == 0:
            if len(doc) > 0:
                words, tags = zip(*doc)
                yield (list(words), list(tags))
            doc = []
            
def transform_to_dataset(tagged_sentences):
    X, y = [], []
    for words, tags in tagged_sentences:
        for index, word  in enumerate(words):
            X.append(features(words, index, tags))
            y.append(tags[index])
    return X, y
    
def transform_to_dataset_inference(tagged_sentences):
    X, y = [], []
    for words, tags in tagged_sentences:
        for index, word  in enumerate(words):
            X.append(features(words, index, tags))
            y.append(tags[index])
    return X, y
            
def evaluation(TEST_DATA):
    y_pred, y_true = [], []
    for words, tags in TEST_DATA:
        for i, (word, pos) in enumerate(pos_tag(words)):
            y_pred.append(pos)
            y_true.append(tags[i])
    return y_pred, y_true
    
def pos_tag(sentence):
    feat = features(sentence, index)
    
    
    
    
    
    
    
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    
    return zip(sentence, tags)
    
    

training_sentences = list(gen_corpus(train_path))
X, y = transform_to_dataset(training_sentences)
clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=True)),
    ('classifier',  LogisticRegression(n_jobs=4, max_iter=200, verbose=True))
])
clf.fit(X, y)


import pickle
pickle.dump((clf, list(Universal_tag_set)), open("log_regression.p", "wb"))

test_sentences = list(gen_corpus(test_path))
X_test, y_test = transform_to_dataset(test_sentences)

print( "Accuracy:", clf.score(X_test, y_test))


y_pred, y_true = evaluation(test_sentences)
for l in classification_report(y_true, y_pred).split('\n'):
    print(l)
    
t = "今天天气非常好。"
print(list(pos_tag(t)))
