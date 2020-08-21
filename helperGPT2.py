from transformers import GPT2Tokenizer
import numpy as np
import pandas as pd
import timeit
import torch
import transformers
assert(transformers.__version__ == '2.6.0')

tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
special_tokens = {'bos_token': '<|startoftext|>', 'eos_token': '<|endoftext|>',
                  'pad_token': '<pad>', 'additional_special_tokens': ['<|keyword|>', '<|summarize|>']}
tokenizer.add_special_tokens(special_tokens)
assert(len(tokenizer) == 50261)


def write_input_ids(word_batch, max_len=1024):
    """
    HELPER FUNCTION
    input: batch of keywords, and text
    input: maximum length of text cap at 1024 for GPT2

    output: list of input_tokens with 4 elements, each with len = max_len
    """
    key, abstract, abstract2, dis1, dis2 = word_batch

    token_key = tokenizer.encode('<|startoftext|> ') + tokenizer.encode(
        key, max_length=190) + tokenizer.encode(' <|summarize|>')

    token_abstract = tokenizer.encode(
        abstract, max_length=tokenizer.max_len - 201)
    token_abstract_shuffle = tokenizer.encode(
        abstract2, max_length=tokenizer.max_len - 201)
    np.random.shuffle(token_abstract_shuffle)
    token_dis1 = tokenizer.encode(dis1, max_length=tokenizer.max_len - 201)
    token_dis2 = tokenizer.encode(dis2, max_length=tokenizer.max_len - 201)

    input_true = token_key + token_abstract + \
        tokenizer.encode(' <|endoftext|>')
    input_dis1 = token_key + token_abstract_shuffle + \
        tokenizer.encode(' <|endoftext|>')
    input_dis2 = token_key + token_dis1 + tokenizer.encode(' <|endoftext|>')
    input_dis3 = token_key + token_dis2 + tokenizer.encode(' <|endoftext|>')

    if max_len == None:
        max_len = max(len(input_true), len(input_dis1),
                      len(input_dis2), len(input_dis3))
    for i in [input_true, input_dis1, input_dis2, input_dis3]:
        while len(i) < max_len:
            i.append(tokenizer.pad_token_id)
    list_input_token = [input_true, input_dis1, input_dis2, input_dis3]

    return list_input_token


def write_token_type_ids(list_input_ids, max_len=1024):
    """
    HELPER FUNCTION
    input: list of input tokens (generated from the function write_input_ids)
    input: maximum length of text (must be the same as argument in the write_input_ids function)
    output: list of token type ids (list of 4 elements each has tokens = max_len)
    """
    list_segment = []
    for item in list_input_ids:
        try:
            item.index(tokenizer.eos_token_id)
        except:
            item[-1] = tokenizer.eos_token_id
        num_seg_a = item.index(tokenizer.additional_special_tokens_ids[1]) + 1
        end_index = item.index(tokenizer.eos_token_id)
        num_seg_b = end_index - num_seg_a + 1
        num_pad = max_len - end_index - 1
        segment_ids = [tokenizer.additional_special_tokens_ids[0]]*num_seg_a + \
            [tokenizer.additional_special_tokens_ids[1]] * \
            num_seg_b + [tokenizer.pad_token_id]*num_pad
        list_segment.append(segment_ids)
    return list_segment


def write_lm_labels(list_input_ids, list_type_label):
    """
    HELPER FUNCTION
    input: list of input tokens (generated from the function write_input_ids)
    input: list of token type ids (generated from the write_token_type_ids)
    output: list of language modeling expected output (list of 4 element each has tokens = max_len)
    notice that the token = -100 is used as mask for non-answer output
    """
    lm_result = list(
        map(lambda x: -100 if int(x[1]) != 50260 else x[0], zip(list_input_ids[0], list_type_label[0])))
    list_lm_label = []
    list_lm_label.append(lm_result)

    for item in list_input_ids[1:]:
        list_lm_label.append([-100]*len(item))

    return list_lm_label


def write_mc_labels():
    """
    HELPER FUNCTION
    write the multiple choice label

    output: list of 4 elements where 1 = correct multiple choice output and 0 for incorrect
    """
    return [1, 0, 0, 0]


def write_last_token(list_input_ids):
    """
    HELPER FUNCTION
    write the last token label
    the last token will be used for the multiple choice selection

    input: list of input tokens (generated from the function write_input_ids)
    output: list of 4 tokens i.e. the index of the <|endoftext|> tokens of each input_ids
    """
    return list(map(lambda x: x.index(tokenizer.eos_token_id), list_input_ids))


def shuffle_batch(list_input_ids, list_type_labels, list_last_tokens, list_lm_labels, list_mc_labels):
    """
    HELPER FUNCTION
    we use this function to shuffle the multiple choices
    Return the numpy array of the input token, type label, last token, lm_label, and mc_label
    """
    array_input_token = np.array(list_input_ids)
    array_segment = np.array(list_type_labels)
    array_mc_token = np.array(list_last_tokens)
    array_lm_label = np.array(list_lm_labels)
    array_mc_label = np.array(list_mc_labels)

    randomize = np.arange(4)
    np.random.shuffle(randomize)

    array_input_token = array_input_token[randomize]
    array_segment = array_segment[randomize]
    array_mc_token = array_mc_token[randomize]
    array_lm_label = array_lm_label[randomize]
    array_mc_label = array_mc_label[randomize]

    return (array_input_token, array_segment, array_mc_token, array_lm_label, array_mc_label)


def write_torch_tensor(np_batch):
    """
    HELPER FUNCTION

    convert the numpy array to pytorch tensor (longtensor type)
    """
    torch_input_token = torch.tensor(
        np_batch[0], dtype=torch.long).unsqueeze(0)
    torch_segment = torch.tensor(np_batch[1], dtype=torch.long).unsqueeze(0)
    torch_last_token = torch.tensor(np_batch[2], dtype=torch.long).unsqueeze(0)
    torch_lm_label = torch.tensor(np_batch[3], dtype=torch.long).unsqueeze(0)
    torch_mc_label = torch.tensor(
        [np.argmax(np_batch[4])], dtype=torch.long).unsqueeze(0)
    return (torch_input_token, torch_segment, torch_last_token, torch_lm_label, torch_mc_label)


def execute_tokenization(word_tuple):
    """
    HELPER FUNCTION
    Execute all functions for each sample
    return the tuple of all the tensors
    """
    list_input_ids = write_input_ids(word_tuple)
    list_type_labels = write_token_type_ids(list_input_ids)
    list_last_tokens = write_last_token(list_input_ids)
    list_lm_label = write_lm_labels(list_input_ids, list_type_labels)
    list_mc_labels = write_mc_labels()
    np_tuple = shuffle_batch(list_input_ids, list_type_labels,
                             list_last_tokens, list_lm_label, list_mc_labels)
    tensor_tuple = write_torch_tensor(np_tuple)
    return tensor_tuple
