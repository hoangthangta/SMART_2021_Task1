from bert_classification import *
from read_write_file import *
from question_classification import *
import random

import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

# prepare dataset............................
dataset = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//task1_wikidata_train.json')
#dataset2 = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//task1_wikidata_train_extension.json', False)
#dataset = dataset + dataset2
#............................................

mode = 'train'
classifier_type = 'type'

if (mode == 'train'):
    #category_by_question_plot(dataset)
    #dataset_trained = rebalance_dataset_by_baseline_method(dataset)
    dataset_trained = dataset
    write_list_to_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//dataset_trained.json', dataset_trained, 'w')
    class_names = sorted(list(set([item[0] for item in type_by_question(dataset_trained, classifier_type)])), key = lambda x: x)
    print('class_names before training: ', class_names)
    write_list_to_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//class_names.json', class_names, 'w')
    train_bert_model(dataset_trained, class_names, pretrained_model = 'bert-base-cased', saved_model_file = 'best_model_state.bin',
                     saved_history_file = 'history_file.json', classifier_type = classifier_type)
else:
    dataset_trained = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//dataset_trained.json')
    class_names = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//class_names.json')
    class_names = [c[0] for c in class_names]   
    validate_dataset(dataset_trained, class_names, pretrained_model = 'bert-base-cased', saved_model_file = 'best_model_state.bin',
                     saved_history_file = 'history_file.json', classifier_type = classifier_type)

# predict a single question .......................................
'''pretrained_model = 'bert-base-cased'
model_file_name = 'best_model_state.bin'
class_names = ['resource', 'literal', 'boolean']
question = 'How many?'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
model = CategoryClassifier(len(class_names), pretrained_model)
model.load_state_dict(torch.load(model_file_name))
predict_single_question(question, tokenizer, model)'''
# .................................................................
#tokenizer = BertTokenizer.from_pretrained(pretrained_model)
#get_bert_tokenizer(dataset, tokenizer, out_file_name = 'bert_embeddings.json')
