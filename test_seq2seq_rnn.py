from seq2seq_rnn import *

mode = 'train'
corpus = 'wikidata'

if (mode == 'train'):
    train_seq2seq_model(n_iters = 100000,
                        dataset_file = corpus + '//task1_' + corpus + '_rare_resource_duplicate.json',
                        corpus = corpus)

elif(mode == 'validate'):

    encoder_file = 'seq2seq_encoder_' + corpus + '.dict'
    decoder_file  = 'seq2seq_decoder_' + corpus + '.dict'
    
    encoder, decoder, input_lang, output_lang = load_seq2seq_model(encoder_file = encoder_file,
                                                                   decoder_file = decoder_file,
                                                                   corpus = corpus)

    sentence = 'Where was Apple born'
    output = predict_single_sentence(encoder, decoder, input_lang, output_lang, sentence)
    output = output.replace('<EOS>', '')
    print('question: ', sentence)
    print('question_type: ', output)

elif(mode == 'predict'):
    
    # rare type prediction, should use with test_bert_classification.py, not HERE!
    dataset = load_list_from_json_file(corpus + '//task1_' + corpus + '_test_pred.json', True)

    encoder_file = 'seq2seq_encoder_' + corpus + '.dict'
    decoder_file  = 'seq2seq_decoder_' + corpus + '.dict'
    predict_dataset_seq2seq(dataset, corpus, encoder_file, decoder_file,
                            out_file_name = corpus + '//task1_' + corpus + '_test_pred.json')

