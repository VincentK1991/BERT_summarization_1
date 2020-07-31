# -*- coding: utf-8 -*-

"""# code here"""

import numpy as np
import timeit
import torch
#from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import json, argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import transformers
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
print('use transformers version = ',transformers.__version__) # make sure it is 2.6.0

def main(args):
  """
  """

  model = GPT2DoubleHeadsModel.from_pretrained(args.model_directory)
  tokenizer = GPT2Tokenizer.from_pretrained(args.model_directory)
  
  # Add a [CLS] to the vocabulary (we should train it also!)
  special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
  tokenizer.add_special_tokens(special_tokens)
  print('total length of vocab should be 50261 = ', len(tokenizer))
  model.resize_token_embeddings(len(tokenizer))
  print(' ')


  model = model.to(device)
  input_text = '<|startoftext|> ' + args.input_file +' <|summarize|>'
  input_token = tokenizer.encode(input_text)
  input_token_torch = torch.tensor(input_token, dtype=torch.long)

  generated_output = model.generate(
      input_ids=input_token_torch.unsqueeze(0).to(device),
      max_length=args.max_length + len(input_token),
      min_length = args.min_length + len(input_token),
      temperature=args.temperature,
      decoder_start_token_id= '<|summarize|>',
      top_k=args.top_k,
      top_p=args.top_p,
      repetition_penalty=None,
      do_sample=True,
      num_return_sequences=args.num_return_sequences)
  batch_answer = []
  for item in generated_output:
    batch_answer.append(tokenizer.decode(item[len(input_token):],skip_special_tokens=True))
  f = open("summary.txt","a")
  f.writelines(batch_answer)
  f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT2 generate summary')

    parser.add_argument('--input_file', type=str,
                        help='text input file (.txt) to be summarized')
    parser.add_argument('--model_directory', type=str, default = None, 
                        help='path to the GPT2 model directory (must contain pytorch_model.bin, config.json, vocab.json, and merges.txt)')
    
    parser.add_argument('--max_length',type=int,default=150,help='maximum token length to generate')
    parser.add_argument('--min_length',type=int,default=50,help='minimum token length to generate')
    parser.add_argument('--top_k',type=int,default=20,help='top k token candidates maximum to consider')
    parser.add_argument('--top_p',type=float,default=0.8,help='cumulative probability of token words to consider as a set of candidates')

    parser.add_argument('--num_return_sequences',type=int,default=3,help='number of sequences to sample')
    parser.add_argument('--temperature',type=float,default=1.0,help='temperature scaling factor for the likelihood distribution')

    main(parser.parse_args())
