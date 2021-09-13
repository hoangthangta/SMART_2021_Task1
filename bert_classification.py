# Bert classification
# This code is followed the article: "Sentiment Analysis with BERT and Transformers by Hugging Face using PyTorch and Python"
# with some adaptations

from read_write_file import *
import random
import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch import nn, optim, functional as F
from torch.utils.data import Dataset, DataLoader
import gc


# configuration .............................
MAX_LEN = 192 # 160 or 256 can be considered
BATCH_SIZE = 8 # should not be larger 16 due to the low RAM
RANDOM_SEED = 42
EPOCHS = 10
#rcParams['figure.figsize'] = 12, 8
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#............................................

class PreparedDataset():
    def __init__(self, sentences, categories, tokenizer, max_len):
        self.sentences = sentences
        self.categories = categories
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = str(self.sentences[item])
        category = self.categories[item]
        encoding = self.tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

        return {
            'sentence': sentence,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'categories': torch.tensor(category, dtype=torch.long)
            }

class CategoryClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model = 'bert-base-cased'):
        super(CategoryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, return_dict=False):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict)
        output = self.drop(pooled_output)
        return self.out(output)

def load_dataset(data_list, class_names, rate=0.8):

    # shuffle dataset
    random.shuffle(data_list)
    first_len = int(len(data_list)*rate)
    second_len = len(data_list) - first_len

    # train set
    train_ds =  data_list[0: first_len]
    train_ds = raw_train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # test set
    test_ds = data_list[first_len:]
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # class names
    # class_names = ['resource', 'literal', 'boolean']

    # create val_ds
    random.shuffle(data_list) # shuffle again
    first_len = int(len(data_list)*rate)
    val_ds = data_list[0: first_len]
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds, val_ds, class_names


def search_type_index_from_class_names(type_list, class_names):

    #print('type_list: ', type_list)
    #print('class_names: ', class_names)
    
    for index, c in enumerate(class_names):
        c_list = c
        if (type(c) is tuple): c_list = [*c]
        else: c_list = [c_list]
        #print('-- c_list: ', c_list)
        n_intersection = len(set(c_list).intersection(set(type_list)))
        #print('-- n_intersection: ', n_intersection)
        if (n_intersection == len(c_list) and len(c_list) == len(type_list)): return index

    return -1

def create_data_loader(dataset, tokenizer, class_names, max_len, batch_size, classifier_type = 'type'):

    sentences = []
    categories = []

    #print('class_names: ', class_names)
    for item in dataset:
        
        item_classifier = []
        if (classifier_type == 'category'): item_classifier = [item[classifier_type]]
        else: item_classifier = item[classifier_type]

        #print('item_classifier: ', item_classifier)
        index = search_type_index_from_class_names(item_classifier, class_names)
        #print('-- index: ', index)
        if (index == -1): continue

        categories.append(index)
        sentences.append(item['question'])

    print('categories: ', categories)
    
    ds = PreparedDataset(sentences=np.array(sentences),
                         categories=np.array(categories),
                         tokenizer=tokenizer,
                         max_len=max_len)
    
    return DataLoader(ds, batch_size=batch_size, num_workers=4)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):

    model = model.train()
    losses = []
    correct_predictions = 0
    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        categories = d["categories"].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, categories)
        correct_predictions += torch.sum(preds == categories)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            categories = d["categories"].to(device)
            # print('categories: ', categories) -1 ???
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, categories)
            correct_predictions += torch.sum(preds == categories)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples, np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()
    sentences = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:
            texts = d["sentence"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            categories = d["categories"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            sentences.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            real_values.extend(categories)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return sentences, predictions, prediction_probs, real_values


def get_single_bert_tokenizer(sentence, tokenizer):

    encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )
    
    return encoding['input_ids'].flatten().tolist()

def convert_dataset_to_bert_tokenizer(dataset, tokenizer, out_file_name = 'bert_tokenizers.json'):

    #tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    #pretrained_model = 'bert-base-cased'
    for item in dataset:
        item_dict = item
        sentence = item['question']
        encoding = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            max_length=MAX_LEN,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
            )

        #print(encoding['input_ids'].flatten().tolist())
        #print('---------------------')
        item_dict['encoding'] =  encoding['input_ids'].flatten().tolist()
        write_single_dict_to_json_file(out_file_name, item_dict)


def train_bert_model(dataset, class_names = [], pretrained_model = 'bert-base-cased' ,
                     saved_model_file = 'best_model_state.bin', saved_history_file = 'history_file.json',
                     classifier_type = 'category', threshold_val_acc = 0.96):

    # prepare dataset
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
 
    df_train, df_test = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    
    print('df_train, df_test: ', len(df_train), len(df_test))
    print('df_val, df_test: ', len(df_val), len(df_test))

    train_data_loader = create_data_loader(df_train, tokenizer, class_names, MAX_LEN, BATCH_SIZE, classifier_type)
    val_data_loader = create_data_loader(df_val, tokenizer, class_names, MAX_LEN, BATCH_SIZE, classifier_type)
    test_data_loader = create_data_loader(df_test, tokenizer, class_names, MAX_LEN, BATCH_SIZE, classifier_type)

    # create model
    model = CategoryClassifier(len(class_names), pretrained_model)
    model = model.to(device)

    data = next(iter(train_data_loader))
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    #F.softmax(model(input_ids, attention_mask), dim=1)
    
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
        )
    loss_fn = nn.CrossEntropyLoss().to(device)

    history = defaultdict(list)
    best_accuracy = 0

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch + 1}/{EPOCHS}')
        print('-' * 10)

        train_acc, train_loss = train_epoch(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(df_train))
        print(f'Train loss {train_loss} accuracy {train_acc}')
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(df_val))

        print(f'Val loss {val_loss} accuracy {val_acc}')
        print()
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
      
        if val_acc > best_accuracy:
            torch.save(model.state_dict(), saved_model_file)
            best_accuracy = val_acc

        # save history step
        print('val_acc: ', val_acc)
        print('train_acc: ', train_acc)

        history_dict = {}
        history_dict['train_acc'] = train_acc.item()
        history_dict['train_loss'] = train_loss
        history_dict['val_acc'] = val_acc.item()
        history_dict['val_loss'] = val_loss
        write_single_dict_to_json_file(saved_history_file, history_dict)

        #if (val_acc.item() > threshold_val_acc): break
        
        torch.cuda.empty_cache()
        gc.collect()
    

def rebalance_dataset_by_baseline_method(dataset):

    final_dataset = []
    # shuffle dataset
    random.shuffle(dataset)

    # get questions by categories
    dataset_by_category_dict = {}
    for item in dataset:
        if (item['category'] not in dataset_by_category_dict):
            dataset_by_category_dict[item['category']] = [item]
        else:
            dataset_by_category_dict[item['category']].append(item)

    # rebalance dataset (remove redudant items)
    min_length = len(dataset)
    for k, v in dataset_by_category_dict.items():
        if (len(v) < min_length): min_length = len(v)

    print('min_length :', min_length)
    for k, v in dataset_by_category_dict.items(): final_dataset += v[:min_length]

    #print('final_dataset: ', len(final_dataset))
    return final_dataset
    

def show_confusion_matrix(confusion_matrix):

    hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('True sentiment')
    plt.xlabel('Predicted sentiment');
    
 
def validate_dataset(dataset, class_names = [],
                     pretrained_model = 'bert-base-cased', saved_model_file = 'best_model_state.bin',
                     saved_history_file = 'history_file.json', classifier_type = 'category'):
    
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    df_train, df_test = train_test_split(dataset, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)
    
    train_data_loader = create_data_loader(df_train, tokenizer, class_names, MAX_LEN, BATCH_SIZE, classifier_type)
    val_data_loader = create_data_loader(df_val, tokenizer, class_names, MAX_LEN, BATCH_SIZE, classifier_type)
    test_data_loader = create_data_loader(df_test, tokenizer, class_names, MAX_LEN, BATCH_SIZE, classifier_type)

    loss_fn = nn.CrossEntropyLoss().to(device)

    model = CategoryClassifier(len(class_names), pretrained_model)
    model.load_state_dict(torch.load(saved_model_file))
    model = model.to(device)
    
    test_acc, _ = eval_model(model, test_data_loader, loss_fn, device, len(df_test))
    test_acc.item()

    y_texts, y_pred, y_pred_probs, y_test = get_predictions(model, test_data_loader)
    print('y_pred: ', y_pred)
    print('y_test: ', y_test)
    print('class_names: ', class_names)
    
    print(classification_report(y_test, y_pred, target_names=class_names))

    # confusion matrix
    '''cm = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
    show_confusion_matrix(df_cm)'''


def predict_single_question(question, tokenizer, model):

    encoded_question = tokenizer.encode_plus(question,
                                           max_length=MAX_LEN,
                                           add_special_tokens=True,
                                           return_token_type_ids=False,
                                           pad_to_max_length=True,
                                           return_attention_mask=True,
                                           return_tensors='pt',
                                           )

    input_ids = encoded_question['input_ids'].to(device)
    attention_mask = encoded_question['attention_mask'].to(device)
    output = model(input_ids, attention_mask)
    _, prediction = torch.max(output, dim=1)
    print(f'Question: {question}')
    print(f'Category: {class_names[prediction]}')




