import re
import nltk
from nltk.corpus import stopwords


class LogTokenizer:
    def __init__(self):
        self.word2index = {'[PAD]': 0, '[CLS]': 1, '[MASK]': 2}
        self.index2word = {0: '[PAD]', 1: '[CLS]', 2: '[MASK]'}
        self.n_words = 3  # Count SOS and EOS
        self.stop_words = set(stopwords.words('english'))
        self.regextokenizer = nltk.RegexpTokenizer('\w+|.|')

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def tokenize(self, sent):
        sent = re.sub(r'/.*:', '', sent, flags=re.MULTILINE)
        sent = re.sub(r'/.*', '', sent, flags=re.MULTILINE)
        sent = self.regextokenizer.tokenize(sent)
        sent = [w.lower() for w in sent]
        sent = [word for word in sent if word.isalpha()]
        sent = [w for w in sent if w not in self.stop_words]
        sent = ['[CLS]'] + sent
        for w in range(len(sent)):
            self.add_word(sent[w])
            sent[w] = self.word2index[sent[w]]
        return sent

    def convert_tokens_to_ids(self, tokens):
        return [self.word2index[w] for w in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.index2word[i] for i in ids]
