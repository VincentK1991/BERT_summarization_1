from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
import timeit
import torch

from torch.utils.data import TensorDataset
import transformers
import json
import argparse
import nltk

from helperGPT2 import execute_tokenization
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
assert(transformers.__version__ == '2.6.0')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')

special_tokens = {'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>',
                  'pad_token': '<pad>', 'additional_special_tokens': ['<|keyword|>', '<|summarize|>']}
tokenizer.add_special_tokens(special_tokens)
assert(len(tokenizer) == 50261)

"""
=================================================
END OF IMPORT AND INITIALIZATION
START OF THE HELPFER FUNCTION SECTION
=================================================
"""


def load_words(df, num, keep=0.8, with_title=False):
    """import dataframe with number of what sample to choose,
    return a keyword (together with title or not) as strings
    and abstract (gold label for summarization).
    and 3 distractors. all as a tuple of 5 strings"""
    arr_distract = np.random.randint(len(df), size=2)
    keyword = df['keyword_POS'][num]
    if with_title:
        title = nltk.word_tokenize(df['title'][num])
        keyword = title + keyword

    keyword_list = np.array(keyword)
    len_keyword = len(keyword_list)
    list_len = [i for i in range(len_keyword)]
    list_len = np.sort(np.random.choice(
        list_len, int(len_keyword*keep) + 1, replace=False))
    keyword_list = list(keyword_list[list_len])

    keyword = ' '.join(keyword_list)
    abstract = df['abstract'][num]
    abstract2 = df['abstract'][num]
    distract1 = df['abstract'][arr_distract[0]]
    distract2 = df['abstract'][arr_distract[1]]

    return (keyword, abstract, abstract2, distract1, distract2)


def tag_pull_abstract(df, list_POS):
    """ return list of keyword list
    input: pandas dataframe
                    list of part of speech tag (in order to generate keyword)
    ourput: List(List(keyword string))"""
    list_tokenized = df['abstract'].apply(
        lambda x: nltk.pos_tag(nltk.word_tokenize(x))).values
    list_answer = [[item[0] for item in row if item[1] in list_POS]
                   for row in list_tokenized]
    #list_answer = list(map(lambda x: ' '.join(x), list_answer))
    return list_answer


"""
======================================
  END OF HELPER FUNCTION SECTION
======================================
"""


def main(args):
    """
    execute the pre-processing given the arguments in CLI
    write
        the tensor dataset in pytorch .pt file
    """
    print('read the input csv file')
    df = pd.read_csv(args.input)
    df = df.dropna(subset=['abstract']).reset_index(drop=True)
    list_POS = ['FW', 'JJ', 'NN', 'NNS', 'NNP',
                'VB', 'VBD', 'VBG', 'VBN', 'VBZ', 'VBP']

    start = timeit.default_timer()
    print('create the keyword tags')
    list_keys = tag_pull_abstract(df, list_POS)
    df['keyword_POS'] = list_keys
    stop = timeit.default_timer()

    df = df.dropna(subset=['keyword_POS']).reset_index(drop=True)
    print('finished keyword tagging in {:.3f} sec'.format(stop - start))
    print('input sample size = ', df.shape[0])
    print(' ')

    print('pre-initializing the tensors')
    tensor_ids = torch.zeros(df.shape[0], 4, 1024)
    tensor_type = torch.zeros(df.shape[0], 4, 1024)
    tensor_last_token = torch.zeros(df.shape[0], 4)
    tensor_lm = torch.zeros(df.shape[0], 4, 1024)
    tensor_mc_label = torch.zeros(df.shape[0], 1)

    print('start-tokenizing-text')
    print('this might take a while ... go get a cup of coffee...')
    start = timeit.default_timer()
    for index in range(df.shape[0]):
        word_tuple = load_words(df, index)
        tensor_batch = execute_tokenization(word_tuple)
        tensor_ids[index] = tensor_batch[0]
        tensor_type[index] = tensor_batch[1]
        tensor_last_token[index] = tensor_batch[2]
        tensor_lm[index] = tensor_batch[3]
        tensor_mc_label[index] = tensor_batch[4]

        if index % 1000 == 0:
            stop = timeit.default_timer()
            print('tokenizing {} iterations took {:.3f} sec'.format(
                index, stop - start))
            start = timeit.default_timer()

    print('finished tokenization')
    tensor_dataset = TensorDataset(
        tensor_ids, tensor_type, tensor_last_token, tensor_lm, tensor_mc_label)
    output_filename = args.output + '.pt'
    torch.save(tensor_dataset, output_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT2 pre-processing')
    parser.add_argument('--input', type=str,
                        help='the raw input data file in csv format')
    parser.add_argument('--output', type=str, default='tensor_dataset',
                        help='the output file name')
    main(parser.parse_args())
