import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, WeightedRandomSampler
from torchvision import transforms, utils
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import trange
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import random
import re
from nltk.corpus import wordnet 
from keras.preprocessing.sequence import pad_sequences

import os

class LogReader:
    def __init__(self, log_format, log_name, indir='./', outdir='./result/', rex=[], every_n=10, max_lines=2000000):
    
        self.path = indir
       
        self.logName = log_name
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex
        self.every_n = every_n
        self.max_lines = max_lines
    def log_to_dataframe(self, log_file, regex, headers, logformat):
            """ Function to transform log file to dataframe 
            """
            log_messages = []
            linecount = 0
            
            if self.max_lines:
                with open(log_file, 'r', encoding="latin-1") as fin:
                    for i,  line in enumerate(fin):
                        if i % self.every_n == 0:
                            try:
                                match = regex.search(line.strip())
                                message = [match.group(header) for header in headers]
                                log_messages.append(message)
                                linecount += 1
                            except Exception as e:
                                pass
                        if i==self.max_lines:
                            break
            else:
                with open(log_file, 'r', encoding="latin-1") as fin:
                    for i,  line in enumerate(fin):
                        if i % self.every_n == 0:
                            try:
                                match = regex.search(line.strip())
                                message = [match.group(header) for header in headers]
                                log_messages.append(message)
                                linecount += 1
                            except Exception as e:
                                pass
            logdf = pd.DataFrame(log_messages, columns=headers)
            logdf.insert(0, 'LineId', None)
            logdf['LineId'] = [i + 1 for i in range(linecount)]
            return logdf


    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex

    def load_data(self):
        headers, regex = self.generate_logformat_regex(self.log_format)
        self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)
        
        
class LogTokenizer:
    def __init__(self):
        self.word2index = {'[PAD]':0, '[CLS]':1, '[MASK]':2}
        self.index2word = {0:'[PAD]', 1:'[CLS]', 2:'[MASK]'}
        self.n_words = 3  # Count SOS and EOS
        self.stop_words = set(stopwords.words('english'))
        self.regextokenizer =  nltk.RegexpTokenizer('\w+|.|')
        
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def tokenize(self, sent):
        sent = re.sub(r'\/.*:', '', sent, flags=re.MULTILINE)
        sent = re.sub(r'\/.*', '', sent, flags=re.MULTILINE)
        sent = self.regextokenizer.tokenize(sent)
        sent = [w.lower() for w in sent]
        sent = [word for word in sent if word.isalpha()]
        sent = [w for w in sent if not w in self.stop_words]
        sent = ['[CLS]'] + sent
        for w in range(len(sent)):
            self.addWord(sent[w])
            sent[w] = self.word2index[sent[w]]
        return sent
    
    def convert_tokens_to_ids(self, tokens):
        return [self.word2index[w] for w in tokens]
    
    def convert_ids_to_tokens(self, ids):
        return [self.index2word[i] for i in ids]

    
def do_syn_ant_augmentation(log_line, synonym_percentag=0.2, antonym_percentag=0.05):
    words = log_line.strip().split()
    num_words = len(words)
    tmp1 = log_line.strip().split()
    if synonym_percentag!=0:  
        synonym_indices = np.random.choice(np.arange(0, num_words), size = int(synonym_percentag * num_words) 
                                       if int(synonym_percentag * num_words) > 0 else 1).tolist()
    
        syn = list()
        for si in synonym_indices:
            for synset in wordnet.synsets(tmp1[si]):
                for lemma in synset.lemmas():
                    syn.append(lemma.name())
            if len(syn) > 0:
                syn = [s for s in syn if "_" not in s] # Filter all underscore words
            # Replace word by random synonym
            if len(syn) > 0:
                words[si] = random.choice(syn)
            syn.clear()
    if antonym_percentag != 0:
        antonym_indices = np.random.choice(np.arange(0, num_words), size = int(antonym_percentag * num_words) 
                                       if int(antonym_percentag * num_words) > 0 else 0)
        ant = list()
        for ai in antonym_indices:
                
            for synset in wordnet.synsets(tmp1[ai]):
                for lemma in synset.lemmas():
                    if lemma.antonyms():    #When antonyms are available, add them into the list
                        ant.append(lemma.antonyms()[0].name())
            if len(ant) > 0:
                tmp=[]
                for a in ant:
                    if "_" in a:
                        tmp.append(' '.join(a.split("_")))
                    else:
                        tmp.append(a)
                ant=tmp
            if len(ant) > 0:
                w = random.choice(ant)
                if words[ai] != w:
                    words[ai] = w

            ant.clear()   
   
    return ' '.join(words)

def get_data(log_file, input_dir, output_dir, log_format, regex=[], every_n=10, aux=0, max_lines=5000000):
    reader = LogReader(log_format, log_file, indir=input_dir, outdir=output_dir, rex=regex, every_n=every_n, max_lines=max_lines)
    reader.load_data()
    log_payload, true_labels = reader.df_log.Content, np.where(reader.df_log.t.values=='-',0,1)
    del reader
    if aux != 0:
        df_anomalies = log_payload.iloc[true_labels.flatten()==1].sample(n=aux).values
        df_normal = log_payload.iloc[true_labels.flatten()==0].sample(n=aux).values
        return df_normal, df_anomalies
    else:
        return log_payload, true_labels
    
def get_data_a(log_file, input_dir, output_dir, log_format, regex=[], every_n=10, aux=0, max_lines=2000000):
    reader = LogReader(log_format, log_file, indir=input_dir, outdir=output_dir, rex=regex, every_n=every_n, max_lines=max_lines)
    reader.load_data()
    log_payload, true_labels = reader.df_log.Content, np.where(reader.df_log.t.values=='FATAL',0,1)
    del reader
    if aux != 0:
        df_anomalies = log_payload.iloc[true_labels.flatten()==1].sample(n=aux).values
        df_normal = log_payload.iloc[true_labels.flatten()==0].sample(n=aux).values
        return df_normal, df_anomalies
    else:
        return log_payload, true_labels
    
def get_padded_data(data, pad_len):
        pd = pad_sequences(data, maxlen=pad_len, dtype="long", 
                             truncating="post", padding="post")
        return pd