from bert_classification import *
from seq2seq_rnn import *

from read_write_file import *

import random

def classifier_by_question(dataset, category_string = 'category_string', split = False, first_item = False):

    type_dict = {}

    for item in dataset:
        temp_type = item[category_string] 
        if (temp_type not in type_dict): type_dict[temp_type] = 1
        else: type_dict[temp_type] +=1
   

    type_dict = sorted(type_dict.items(), key = lambda x: x[1], reverse = True)
    return type_dict

# train......................................
mode = 'pretrain' # train or validate or nothing
classifier_type = 'type_string' # category or type
pretrained_model = 'bert-base-cased' # can try with different BERT models
corpus = 'wikidata' 
dataset = []
#............................................

# prepare dataset............................

if (classifier_type == 'category_string'):
    dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_category.json', False)
    #dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_popular_resource.json', False)
    
elif(classifier_type == 'type_string'):
    # popular vs rare

    if (corpus == 'dbpedia'):
        dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_resource_flatten.json', False)
    else:
        dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_popular_resource.json', False)
    #dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_rare_resource.json', False)

#............................................

if (mode == 'train'):
    write_list_to_json_file(corpus + '//dataset_trained_' + classifier_type + '.json',
                            dataset, 'w')

    #type_dict = classifier_by_question(dataset, classifier_type)
        
    class_names = sorted(list(set([item[0] for item in classifier_by_question(dataset, classifier_type)])), key = lambda x: x)
    class_names = [c.strip() for c in class_names if c.strip() != '']
    print('class_names: ', class_names, len(class_names))

    write_list_to_json_file(corpus + '//class_names_' + classifier_type + '.json', class_names, 'w')

    train_bert_model(dataset, class_names, pretrained_model = pretrained_model,
                     saved_model_file = 'best_bert_model_state_' + classifier_type + '_' + corpus + '.bin',
                     saved_history_file = 'history_file_' + classifier_type + '_' + corpus + '.json',
                     classifier_type = classifier_type)
    
elif(mode == 'validate'):
    dataset = load_list_from_json_file(corpus + '//dataset_trained_' + classifier_type + '.json')
    class_names = load_list_from_json_file(corpus + '//class_names_' + classifier_type + '.json')

    #if (classifier_type == 'type'):
    #class_names = [c for c in class_names]
    
    validate_dataset(dataset, class_names, pretrained_model = pretrained_model,
                     saved_model_file = 'best_bert_model_state_' + classifier_type + '_' + corpus + '.bin',
                     saved_history_file = 'history_file_' + classifier_type + '_' + corpus + '.json',
                     classifier_type = classifier_type)

elif(mode == 'pretrain'):
    dataset = load_list_from_json_file(corpus + '//dataset_trained_' + classifier_type + '.json')
    class_names = load_list_from_json_file(corpus + '//class_names_' + classifier_type + '.json')


    pretrain_bert_model(dataset, class_names, pretrained_model = pretrained_model,
                     saved_model_file = 'best_bert_model_state_' + classifier_type + '_' + corpus + '.bin',
                     saved_history_file = 'history_file_' + classifier_type + '_' + corpus + '.json',
                        classifier_type = classifier_type)

elif(mode == 'predict'):

    # "n\/a" questions
    
    # category prediction
    pretrained_model = 'bert-base-cased'
    classifier_type = 'category_string'
    model_file_name = 'best_bert_model_state_' + classifier_type + '_' + corpus + '.bin'
    class_names = load_list_from_json_file(corpus + '//class_names_' + classifier_type + '.json')
    dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_test.json', True)

    predict_dataset_bert(dataset, class_names, classifier_type = classifier_type,
                    pretrained_model = pretrained_model,
                    model_file_name = model_file_name,
                    out_file_name = corpus + '//task1_' + corpus + '_test_pred.json')

    # popular type prediction
    pretrained_model = 'bert-base-cased'
    classifier_type = 'type_string'
    model_file_name = 'best_bert_model_state_' + classifier_type + '_' + corpus + '.bin'
    class_names = load_list_from_json_file(corpus + '//class_names_' + classifier_type + '.json')
    dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_test_pred.json', True)

    predict_dataset_bert(dataset, class_names, classifier_type = classifier_type,
                    pretrained_model = pretrained_model,
                    model_file_name = model_file_name,
                    out_file_name = corpus + '//task1_' + corpus + '_test_pred.json')

    # dbpedia has no rare types
    #if (corpus == 'dbpedia'): continue
    
    # rare type prediction
    #dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_test_pred.json', True)

    # the prediction test files in "wikidata" and "dbpeadia" folders
    '''encoder_file = 'seq2seq_encoder_' + corpus + '.dict'
    decoder_file  = 'seq2seq_decoder_' + corpus + '.dict'
    predict_dataset_seq2seq(dataset, corpus, encoder_file, decoder_file,
                            out_file_name = corpus + '//task1_' + corpus + '_test_pred.json')'''


elif(mode == 'predict_single_type'):

    pretrained_model = 'bert-base-cased'
    classifier_type = 'type_string'
    model_file_name = 'best_bert_model_state_' + classifier_type + '_' + corpus + '.bin'
    class_names = load_list_from_json_file(corpus + '//class_names_' + classifier_type + '.json')
    dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_test_pred.json', True)
    
    question = 'What is the subsidiary company working for Leonard Maltin?'
    predict_single_type(question, class_names, classifier_type = classifier_type,
                    pretrained_model = pretrained_model,
                    model_file_name = model_file_name,
                    out_file_name = model_file_name)
    
    

