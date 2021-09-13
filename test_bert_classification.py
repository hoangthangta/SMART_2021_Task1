from bert_classification import *
from read_write_file import *
import random

def type_by_question(dataset, classifier_type = 'type', split = False, first_item = False):
    type_dict = {}

    if (split == False):
        for item in dataset:
            #temp_type = '_'.join(i for i in item['type'])

            temp_type = tuple()
            if (classifier_type == 'type'):
                temp_type = tuple(item[classifier_type])
            else:
                temp_type = tuple([item[classifier_type]])
            
            if (temp_type not in type_dict):
                type_dict[temp_type] = 1
            else:
                type_dict[temp_type] +=1
    else:
        for item in dataset:
            if (first_item == True):
                if (item['type'][0] not in type_dict): type_dict[item['type'][0]] = 1
                else:  type_dict[item['type'][0]] += 1
            else:
                for t in item['type']:
                    if (t not in type_dict): type_dict[t] = 1
                    else: type_dict[t] += 1     

    type_dict = dict(sorted(type_dict.items(), key = lambda x: x[1], reverse = True))
    return type_dict

# train......................................
mode = 'train' # train or validate
classifier_type = 'type' # category or type
pretrained_model = 'bert-base-cased' # can try with different BERT models
corpus = 'wikidata' # or dbpedia (important!!!)
#............................................

# prepare dataset............................

if (classifier_type == 'category'):
    dataset = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//task1_wikidata_train.json')
    # oversampling on boolean and literal questions
    #dataset2 = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//task1_wikidata_train_extension.json', False)
    #dataset = dataset + dataset2

else:
    dataset = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//wikidata//task1_wikidata_train_flatten.json', False)
    
#............................................

if (mode == 'train'):
    dataset_trained = dataset
    write_list_to_json_file('smart2021-AT_Answer_Type_Prediction//' + corpus + '//dataset_trained_' + classifier_type + '.json',
                            dataset_trained, 'w')

    type_dict = type_by_question(dataset_trained, classifier_type)
        
    #class_names = [item[0] for item in type_by_question(dataset_trained, classifier_type)]
    class_names = sorted(list(set([item[0] for item in type_by_question(dataset_trained, classifier_type)])), key = lambda x: x)
    class_names = [c.strip() for c in class_names if c.strip() != '']
    
    print('class_names before training: ', class_names, len(class_names))

    write_list_to_json_file('smart2021-AT_Answer_Type_Prediction//' + corpus + '//class_names_' + classifier_type + '.json',
                            class_names, 'w')

    train_bert_model(dataset_trained, class_names, pretrained_model = pretrained_model,
                     saved_model_file = 'best_bert_model_state.bin',
                     saved_history_file = 'history_file_' + classifier_type + '.json', classifier_type = classifier_type)
else:
    dataset_trained = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//' + corpus + '//dataset_trained_'
                                               + classifier_type + '.json')
    
    class_names = load_list_from_json_file('smart2021-AT_Answer_Type_Prediction//' + corpus + '//class_names_'
                                           + classifier_type + '.json')

    #if (classifier_type == 'type'):
    #class_names = [c for c in class_names]
    
    validate_dataset(dataset_trained, class_names, pretrained_model = pretrained_model,
                     saved_model_file = 'best_bert_model_state.bin',
                     saved_history_file = 'history_file_' + classifier_type + '.json', classifier_type = classifier_type)
#............................................

# predict a single question .................
'''pretrained_model = 'bert-base-cased'
model_file_name = 'best_model_state.bin'
class_names = ['resource', 'literal', 'boolean']
question = 'How many?'
tokenizer = BertTokenizer.from_pretrained(pretrained_model)
model = CategoryClassifier(len(class_names), pretrained_model)
model.load_state_dict(torch.load(model_file_name))
predict_single_question(question, tokenizer, model)'''
# .................................................................
