# coding=utf-8

from itertools import chain
import time
import random
import os
import cPickle
import warnings


import nltk
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
import pycrfsuite
import pymorphy2

class SequenceNPFinder(object):
    def __init__(self):
        self.siblings = 2
        self.use_isupper = True
        self.use_istitle = True
        self.use_isdigit = True
        self.use_tokens = False
        self.use_BOS_EOS = True

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1] or ''
        features = [
            'bias',
            'postag=' + postag,
        ]
        if self.use_isupper:
            features.append('word.isupper=%s' % word.isupper())
        if self.use_istitle:
            features.append('word.istitle=%s' % word.istitle())
        if self.use_isdigit:
            features.append('word.isdigit=%s' % word.isdigit())
        if self.use_tokens:
            features.append('word.lower=' + word.lower())
        if self.use_BOS_EOS:
            if i == 0:
                features.append('BOS')
            elif i == len(sent)-1:
                features.append('EOS')

        for j in range(-1, -1 * (self.siblings + 1), -1):
            if i+j >= 0:
                word1 = sent[i+j][0]
                postag1 = sent[i+j][1] or ''
                features.extend([
                    str(j)+':postag=' + postag1,
                ])
                if self.use_isupper:
                    features.append(str(j)+':word.isupper=%s' % word1.isupper())
                if self.use_istitle:
                    features.append(str(j)+':word.istitle=%s' % word1.istitle())
                if self.use_isdigit:
                    features.append(str(j)+':word.isdigit=%s' % word1.isdigit())
                if self.use_tokens:
                    features.append(str(j)+':word.lower=' + word1.lower())
                if self.use_BOS_EOS:
                    if i+j == 0:
                        features.append(str(j)+':BOS')

        for j in range(1, (self.siblings + 1), 1):
            if i+j < len(sent):
                word1 = sent[i+j][0]
                postag1 = sent[i+j][1] or ''
                features.extend([
                    str(j)+':postag=' + postag1,
                ])

                if self.use_isupper:
                    features.append(str(j)+':word.isupper=%s' % word1.isupper())
                if self.use_istitle:
                    features.append(str(j)+':word.istitle=%s' % word1.istitle())
                if self.use_isdigit:
                    features.append(str(j)+':word.isdigit=%s' % word1.isdigit())
                if self.use_tokens:
                    features.append(str(j)+':word.lower=' + word1.lower())
                if self.use_BOS_EOS:
                    if i+j == len(sent)-1:
                        features.append(str(j)+':EOS')

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    @staticmethod
    def sent2labels(sent):
        return [label for token, postag, label in sent]

    @staticmethod
    def sent2tokens(sent):
        return [token for token, postag, label in sent]


    def train(self, sents, model_filename):
        t = time.time()
    
        X_train = [self.sent2features(s) for s in sents]
        y_train = [self.sent2labels(s) for s in sents]
    
        trainer = pycrfsuite.Trainer(verbose=False)
        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)
    
        trainer.set_params({
            # 'c1': 1.0,   # coefficient for L1 penalty
            # 'c2': 1e-3,  # coefficient for L2 penalty
            # 'max_iterations': 200,  # stop earlier
    
            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        
        trainer.train(model_filename)

    def load_model(self, model_filename):
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(model_filename)

    def get_tags(self, sent):
        # assert not (self.tagger is None)
        tags = self.tagger.tag(self.sent2features(sent))
        return tags

    def tag_sent(self, sent):
        if len(sent) > 0:
            if len(sent[0]) == 2:
                return [(word, pos, tag) for (word, pos), tag in zip(sent, self.get_tags(sent))]
            elif len(sent[0]) == 3:
                return [(word, pos, tag) for (word, pos, old_tag), tag in zip(sent, self.get_tags(sent))]

    def get_nps(self, sent):
        nps = []
        current = ""
        for (word, pos), tag in reversed(zip(sent, self.get_tags(sent))):
            if tag == 'I':
                current = word + ' ' + current
            elif tag == "B":
                current = word + ' ' + current
                nps.append(current.rstrip())
                current = ""
            elif tag == 'O':
                pass
        return reversed(nps)

    def get_nps_seq(self, sent):
        nps = []
        current = []
        for (word, pos), tag in reversed(zip(sent, self.get_tags(sent))):
            if tag == 'I':
                current.append((word, pos, tag))
            elif tag == "B":
                current.append((word, pos, tag))
                nps.append(reversed(current))
                current = []
            elif tag == 'O':
                pass
        return reversed(nps)

class ExtSequenceNPFinder(SequenceNPFinder):
    def __init__(self):
        super(ExtSequenceNPFinder, self).__init__()

    def word2features(self, sent, i):
        word = sent[i][0]
        postag = sent[i][1] or ''
        case = sent[i][3] or ''
        rod = sent[i][4] or ''
        numb = sent[i][5] or ''
        features = [
            'bias',
            'postag=' + postag,
        ]
        if self.use_isupper:
            features.append('word.isupper=%s' % word.isupper())
        if self.use_istitle:
            features.append('word.istitle=%s' % word.istitle())
        if self.use_isdigit:
            features.append('word.isdigit=%s' % word.isdigit())
        if self.use_tokens:
            features.append('word.lower=' + word.lower())
        if self.use_BOS_EOS:
            if i == 0:
                features.append('BOS')
            elif i == len(sent)-1:
                features.append('EOS')
        features.append('word.case=' + case)
        features.append('word.rod=' + rod)
        features.append('word.number=' + numb)

        for j in range(-1, -1 * (self.siblings + 1), -1):
            if i+j >= 0:
                word1 = sent[i+j][0]
                postag1 = sent[i+j][1] or ''
                case1 = sent[i+j][3] or ''
                rod1 = sent[i+j][4] or ''
                numb1 = sent[i+j][5] or ''
                features.extend([
                    str(j)+':postag=' + postag1,
                ])
                if self.use_isupper:
                    features.append(str(j)+':word.isupper=%s' % word1.isupper())
                if self.use_istitle:
                    features.append(str(j)+':word.istitle=%s' % word1.istitle())
                if self.use_isdigit:
                    features.append(str(j)+':word.isdigit=%s' % word1.isdigit())
                if self.use_tokens:
                    features.append(str(j)+':word.lower=' + word1.lower())
                if self.use_BOS_EOS:
                    if i+j == 0:
                        features.append(str(j)+':BOS')
                features.append(str(j) + ':word.case=' + case1)
                features.append(str(j) + ':word.rod=' + rod1)
                features.append(str(j) + ':word.number=' + numb1)

        for j in range(1, (self.siblings + 1), 1):
            if i+j < len(sent):
                word1 = sent[i+j][0]
                postag1 = sent[i+j][1] or ''
                case1 = sent[i+j][3] or ''
                rod1 = sent[i+j][4] or ''
                numb1 = sent[i+j][5] or ''
                features.extend([
                    str(j)+':postag=' + postag1,
                ])

                if self.use_isupper:
                    features.append(str(j)+':word.isupper=%s' % word1.isupper())
                if self.use_istitle:
                    features.append(str(j)+':word.istitle=%s' % word1.istitle())
                if self.use_isdigit:
                    features.append(str(j)+':word.isdigit=%s' % word1.isdigit())
                if self.use_tokens:
                    features.append(str(j)+':word.lower=' + word1.lower())
                if self.use_BOS_EOS:
                    if i+j == len(sent)-1:
                        features.append(str(j)+':EOS')
                features.append(str(j) + ':word.case=' + case1)
                features.append(str(j) + ':word.rod=' + rod1)
                features.append(str(j) + ':word.number=' + numb1)

        return features

    def sent2features(self, sent):
        return [self.word2features(sent, i) for i in range(len(sent))]

    @staticmethod
    def sent2labels(sent):
        return [label for token, postag, label, case, rod, numb in sent]

    @staticmethod
    def sent2tokens(sent):
        return [token for token, postag, label, case, rod, numb in sent]

    def tag_sent(self, sent):
        if len(sent) > 0:
            if len(sent[0]) == 6:
                return [(word, pos, tag, case, rod, numb) for (word, pos, old_tag, case, rod, numb), tag in zip(sent, self.get_tags(sent))]


class TextNPFinder(SequenceNPFinder):
    def __init__(self):
        super(TextNPFinder, self).__init__()
        self.morph = pymorphy2.MorphAnalyzer()
        self.pymorphy_to_syntagrus_pos = {
            'NOUN': 'S',
            'ADJF': 'A',
            'ADJS': 'A',
            'COMP': 'ADV',
            'VERB': 'V',
            'INFN': 'V',
            'PRTF': 'V',
            'PRTS': 'V',
            'GRND': 'V',
            'NUMR': 'NUM',
            'ADVB': 'ADV',
            'NPRO': 'S',
            'PRED': 'V',
            'PREP': 'PR',
            'CONJ': 'CONJ',
            'PRCL': 'PART',
            'INTJ': 'INTJ',
            }
        self.pymorphy_to_syntagrus_tag = {
            'LATN': 'NID',
            'PNCT': False,
            'NUMB': 'NUM',
            'ROMN': 'NUM',
            'UNKN': 'NID'
        }


    def pos(self, word):
        tag = self.morph.parse(word)[0].tag
        pos = tag.POS

        if pos:
            return self.pymorphy_to_syntagrus_pos[pos]
        else:
            for key in self.pymorphy_to_syntagrus_tag:
                if key in tag:
                    return self.pymorphy_to_syntagrus_tag[key]



    def tokenize_sent(self, sent):
        words = nltk.word_tokenize(sent)
        return words

    def analyze_sent(self, sent):
        return [(word, self.pos(word)) for word in self.tokenize_sent(sent) if self.pos(word)]
        # return [(word, self.pos(word)) for word in self.tokenize_sent(sent)]

    def get_tags(self, sent):
        return super(TextNPFinder, self).get_tags(self.analyze_sent(sent))

    def tag_sent(self, sent):
        sent_with_pos = self.analyze_sent(sent)
        return [(word, pos, tag) for (word, pos), tag in zip(sent_with_pos, super(TextNPFinder, self).get_tags(sent_with_pos))]

    def get_nps(self, sent):
        sent_with_pos = self.analyze_sent(sent)
        nps = []
        current = ""
        for (word, pos), tag in reversed(zip(sent_with_pos, super(TextNPFinder, self).get_tags(sent_with_pos))):
            if tag == 'I':
                current = word + ' ' + current
            elif tag == "B":
                current = word + ' ' + current
                nps.append(current.rstrip())
                current = ""
            elif tag == 'O':
                pass

        return reversed(nps)

    def get_sent_with_parentheses(self, sent):
        nps = self.get_nps(sent)
        min_index = 0
        for np in nps:
            index = sent.find(np, min_index)
            if not index>=0:
                print sent
                print self.get_tags(sent)
                print np
                print ''
                assert False
            else:
                min_index = index + len(np) + 1
                sent = sent[:index] + '[' + np + '] ' + sent[min_index:]
        return sent.rstrip()









