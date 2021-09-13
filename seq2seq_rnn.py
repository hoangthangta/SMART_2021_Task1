from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import time
import math

import gc

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

from read_write_file import *

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy
from spacy.lemmatizer import Lemmatizer, ADJ, NOUN, VERB

nlp = spacy.load('en_core_web_lg')
nlp.add_pipe(nlp.create_pipe('sentencizer'), before='parser')
lemmatizer = nlp.vocab.morphology.lemmatizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

word2vec_size = 300
SOS_token = 0
EOS_token = 1
#SOS_token = [0]*word2vec_size
#EOS_token = [1]*word2vec_size

MAX_LENGTH = 256
RANDOM_SEED = 42
EPOCHS = 10
teacher_forcing_ratio = 0.5
hidden_size = 256

#word2vec_model = load_pretrained_word2vec('D:/wiki-news-300d-1M.vec', 100000)

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        #self.word2vector = {}

    def addSentence(self, sentence):
        '''doc = nlp(sentence)
        for token in doc: self.addWord(token.text)'''
        for word in sentence.split(' '): self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words

            '''try: self.word2vector[word] = word2vec_model['word']
            except: pass'''
        
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, reverse=False, dataset_file = 'smart2021-AT_Answer_Type_Prediction//wikidata//task1_wikidata_train_flatten.json'):
    print("Reading file...")
    
    pairs = []
    dataset = load_list_from_json_file(dataset_file, False)

    # We do not normalize question texts here!!!
    for item in dataset: pairs.append([item['question'], item['type_string'], item['type']]) # triples
    print('Pairs: ', pairs)

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs] 
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(input_lang, output_lang, pairs):
    
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print('-- input_lang + n_words: ', input_lang.name, input_lang.n_words)
    print('-- output_lang + n_words: ', output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def indexesFromSentence(lang, sentence):
    '''doc = nlp(sentence)
    return [lang.word2index[token.text] for token in doc]'''
    return [lang.word2index[word] for word in sentence.split(' ')]

def vectorsFromSentence(lang, sentence):
    doc = nlp(sentence)
    return [lang.word2vector[token.text] for token in doc]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorsFromPair(input_lang, output_lang, pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def eval_pair(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):

    encoder_hidden = encoder.initHidden()

    #encoder_optimizer.zero_grad()
    #decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    #loss.backward()

    #encoder_optimizer.step()
    #decoder_optimizer.step()

    return loss.item() / target_length

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    
def trainIters(encoder, decoder, n_iters, input_lang, output_lang, pairs,
               input_lang_eval, output_lang_eval, pairs_eval, epochs = EPOCHS, print_every=1000, plot_every=100, learning_rate=0.01):

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # choose random pairs
    '''training_pairs = [tensorsFromPair(input_lang, output_lang, random.choice(pairs))
                      for i in range(n_iters)]'''

    training_pairs = [tensorsFromPair(input_lang, output_lang, pair) for pair in pairs]
    print('training_pairs: ', len(training_pairs))
    
    criterion = nn.NLLLoss()

    #loss_list = []
    best_val_acc = 0
    
    for epoch in range(1, epochs + 1):

        epoch_loss = 0
        epoch_avg = 0

        print('....................................................')
        print('epoch: ', epoch)
        
        for iter in range(1, n_iters + 1):
            
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

            epoch_loss += loss
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('---- %s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        
        val_acc, total_val_loss, val_loss = evaluate_pair_val(pairs_eval, input_lang_eval, output_lang_eval,
                      encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        #val_acc, val_loss = 0, 0
        epoch_avg_loss = epoch_loss/n_iters

        print('-- epoch_loss: ', epoch_loss)
        print('-- epoch_avg_loss: ', epoch_avg_loss)
        print('--- val_acc: ', val_acc)
        print('--- val_loss: ', val_loss)
        
        #loss_list.append({'loss': epoch_loss})

        # save the best model
        if (val_acc > best_val_acc):
            torch.save(encoder.state_dict(), 'seq2seq_encoder.dict')
            torch.save(decoder.state_dict(), 'seq2seq_decoder.dict')
            best_val_acc = val_acc

        # save history  
        item_loss_dict = {
            'epoch': str(epoch),
            'total_loss': epoch_loss,
            'avg_loss': epoch_avg_loss,
            'val_acc': val_acc,
            'total_val_loss': total_val_loss,
            'avg_val_loss': val_loss
            }
        write_single_dict_to_json_file('seq2seq_history.json', item_loss_dict)

        gc.collect()

    #showPlot(plot_losses)

def evaluate(encoder, decoder, input_lang, output_lang, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]

def evaluate_pair_val(pair_val, input_lang, output_lang,
                      encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):

    correct_predictions = 0
    total_val_loss = 0 
    for i in range(len(pair_val)):
        pair = pair_val[i]
        #print('>', pair[0])
        #print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, input_lang, output_lang, pair[0])

        output_sentence = ' '.join(output_words)
        output_sentence = output_sentence.replace('<EOS>', '').strip()
        #print('output_sentence:', output_sentence, output_words)

        pred_type_list = output_sentence.split(',')
        pred_type_list = [p.strip() for p in pred_type_list if p.strip() != 0]
        
        origin_type_list = pair[2]
        origin_type_list = [o.strip() for o in origin_type_list if o.strip() != 0]
        
        #common_items = set(pred_type_list).intersection(set(origin_type_list))

        subset_flag = False
        try:
            subset_flag = set(pred_type_list).issubset(set(origin_type_list))
        except:
            pass
        
        if (subset_flag == True): correct_predictions += 1

        tensor_pair = tensorsFromPair(input_lang, output_lang, pair)
        input_tensor = tensor_pair[0]
        target_tensor = tensor_pair[1]
        loss = eval_pair(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        total_val_loss += loss

        #print('<', output_sentence)
        #print('')
    
    return correct_predictions / len(pair_val), total_val_loss, total_val_loss / len(pair_val)


def evaluate_randomly(encoder, decoder, n=5):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

def train_seq2seq_model(n_iters = 100000, dataset_file = 'smart2021-AT_Answer_Type_Prediction//wikidata//task1_wikidata_train_flatten.json'):

    # read dataset
    input_lang, output_lang, pairs = readLangs('question', 'type_string', reverse=False)

    # split datasets
    pair_train, pair_test = train_test_split(pairs, test_size=0.2, random_state=RANDOM_SEED)
    pair_val, pair_test = train_test_split(pair_test, test_size=0.5, random_state=RANDOM_SEED)

    n_iters = len(pair_train)

    print('pair_train, pair_test: ', len(pair_train), len(pair_test))
    print('pair_val, pair_test: ', len(pair_val), len(pair_test))

    # prepare train data
    input_lang, output_lang, pairs = prepareData(input_lang, output_lang, pair_train)
    print(random.choice(pairs))

    input_lang_eval, output_lang_eval, pairs_eval = prepareData(input_lang, output_lang, pair_val)
    
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, n_iters, input_lang, output_lang, pairs, input_lang_eval, output_lang_eval, pairs_eval)
    

def load_seq2seq_model():

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    encoder.load_state_dict(torch.load('seq2seq_encoder.dict'))
    decoder.load_state_dict(torch.load('seq2seq_decoder.dict'))
    
    evaluate_randomly(encoder1, attn_decoder1)