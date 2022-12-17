import nltk
import pandas as pd
import string
import torch
import torchtext

stemmer = nltk.stem.snowball.SnowballStemmer('english')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getDict(dataPipe):

    data_dict = {
        'Question': [],
        'Answer': []
    }
    
    for _, question, answers, _ in dataPipe:
        data_dict['Question'].append(question)
        data_dict['Answer'].append(answers[0])
        
    return data_dict


def loadDF(path):
    # load data
    train_data, val_data = torchtext.datasets.SQuAD1(path)
    
    # convert dataPipe to dictionary 
    train_dict, val_dict = getDict(train_data), getDict(val_data)
    
    # convert Dictionaries to Pandas DataFrame
    train_df = pd.DataFrame(train_dict)    
    validation_df = pd.DataFrame(val_dict)    
    
    return train_df.append(validation_df)


def prepare_text(sentence):
    # clean text and tokenize it 
    sentence = ''.join([s.lower() for s in sentence if s not in string.punctuation])
    sentence = ' '.join(stemmer.stem(w) for w in sentence.split())
    tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(sentence)

    return tokens


def toTensor(vocab, sentence):
    # convert list of words "sentence" to a torch tensor of indices
    indices = [vocab.word2index[word] for word in sentence.split(' ')]
    indices.append(vocab.word2index[''])
    return torch.Tensor(indices).long().to(device).view(-1, 1)


def getPairs(df):
    # convert df to list of pairs
    temp1 = df['Question'].apply(lambda x: " ".join(x) ).to_list()
    temp2 = df['Answer'].apply(lambda x: " ".join(x) ).to_list()
    return [list(i) for i in zip(temp1, temp2)]


def getMaxLen(pairs):
    max_src = 0 
    max_trg = 0
    
    for p in pairs:
        max_src = len(p[0].split()) if len(p[0].split()) > max_src else max_src
        max_trg = len(p[1].split()) if len(p[1].split()) > max_trg else max_trg
        
    return max_src, max_trg

