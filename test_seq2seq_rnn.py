from seq2seq_rnn import *

# work with question type only
# n_iters: will be the size of the corpus
# dataset_file: the file containing dataset (dbpedia or wikidata flatten version)
train_seq2seq_model(n_iters = 100000, dataset_file = 'smart2021-AT_Answer_Type_Prediction//wikidata//task1_wikidata_train_flatten.json')


