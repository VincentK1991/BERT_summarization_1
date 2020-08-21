# -*- coding: utf-8 -*-
"""# code here"""

import numpy as np
import timeit
import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import json, argparse
from transformers import get_linear_schedule_with_warmup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import transformers
from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, AdamW
print('use transformers version = ',transformers.__version__) # make sure it is 2.6.0

def main(args):
  """
  executing the training given the arguments in CLI
  output:
    write pytorch model file, and config files
    write training and validation statistics (in .json)
  """
  train_dict = {'lm_loss':[],'mc_loss':[],'total_loss':[]}
  val_dict = {'lm_loss':[],'mc_loss':[],'total_loss':[]}

  if args.model_directory == None:
    model = GPT2DoubleHeadsModel.from_pretrained('distilgpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
    print('total length of vocab should be 50261 = ', len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    print('resize the model embedding layer')
  else:
    model = GPT2DoubleHeadsModel.from_pretrained(args.model_directory)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_directory)
    special_tokens = {'bos_token':'<|startoftext|>','eos_token':'<|endoftext|>','pad_token':'<pad>','additional_special_tokens':['<|keyword|>','<|summarize|>']}
    print('total length of vocab should be 50261 = ', len(tokenizer))

  # Add a [CLS] to the vocabulary (we should train it also!)
  print(' ')

  train_dataset = torch.load(args.train_data)
  train_sampler = RandomSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
  print('finished downloading train dataset')

  val_dataset = torch.load(args.val_data)
  val_sampler = RandomSampler(val_dataset)
  val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=1)
  print('finished downloading vallidation dataset')

  model = model.to(device)
  optimizer = AdamW(model.parameters(),lr=args.learning_rate,eps=args.eps, correct_bias=True)
  total_steps = len(train_dataloader)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = args.scheduler_warmup, num_training_steps = total_steps)


  for epoch in range(args.epochs):
    start = timeit.default_timer()
    start_iter = timeit.default_timer()
    for iterations,batch in enumerate(train_dataloader):
      lm_loss, mc_loss, total_loss = train(args,batch,iterations,model,optimizer,scheduler)
      train_dict['lm_loss'].append(lm_loss)
      train_dict['mc_loss'].append(mc_loss)
      train_dict['total_loss'].append(total_loss)
      if iterations % args.print_every == 0:
        stop_iter = timeit.default_timer()
        print("Trainer Results - epoch {} - LM loss: {:.2f} MC loss: {:.2f} total loss: {:.2f} report time: {:.1f} sec"
        .format(iterations, train_dict['lm_loss'][-1], train_dict['mc_loss'][-1], train_dict['total_loss'][-1],stop_iter - start_iter))
        start_iter = timeit.default_timer()

    print('end-of-training-epoch')
    stop = timeit.default_timer()
    print("Trainer Results - epoch {} - LM loss: {:.2f} MC loss: {:.2f} total loss: {:.2f} report time: {:.1f} sec"
    .format(epoch, train_dict['lm_loss'][-1], train_dict['mc_loss'][-1], train_dict['total_loss'][-1],stop - start))
    print(' ')
    for iterations,batch in enumerate(val_dataloader):
      lm_loss, mc_loss, total_loss = evaluate(args,batch,model)
      val_dict['lm_loss'].append(lm_loss)
      val_dict['mc_loss'].append(mc_loss)
      val_dict['total_loss'].append(total_loss)

    print('end-of-validation-epoch')
    stop_eval = timeit.default_timer()
    print("Evaluator Results - epoch {} - LM loss: {:.2f} MC loss: {:.2f} total loss: {:.2f} report time: {:.1f} sec"
    .format(epoch, val_dict['lm_loss'][-1], val_dict['mc_loss'][-1], val_dict['total_loss'][-1],stop_eval - stop))
    print(' ')
  model.config.to_json_file(args.model_name + '/config.json')
  tokenizer.save_vocabulary(args.model_name)
  model_file = args.model_name + '/pytorch_model.bin'
  torch.save(model.state_dict(), model_file)
  with open(args.model_name + '/training_loss_' + str(args.epochs) + '_epoch.json', 'w') as fp:
    json.dump(train_dict, fp)
  with open(args.model_name + '/validation_loss_' + str(args.epochs) + '_epoch.json', 'w') as fq:
    json.dump(val_dict, fq)

def train(args,batch,iterations,model,optimizer,scheduler):
  """
  """
  model.train()
  batch = (item.to(device) for item in batch)
  input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
  outputs = model(input_ids = input_ids, mc_token_ids = mc_token_ids, mc_labels = mc_labels,
                  lm_labels = lm_labels, token_type_ids = token_type_ids)
  lm_loss, mc_loss = outputs[0], outputs[1]
  total_loss = lm_loss * args.lm_coef + mc_loss * args.mc_coef
  total_loss = total_loss / args.grad_accumulation
  total_loss.backward()
  torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
  if iterations% args.grad_accumulation == 0:
    optimizer.step()
    optimizer.zero_grad()
  scheduler.step()
  return lm_loss.item(),mc_loss.item(),total_loss.item()*args.grad_accumulation

def evaluate(args,batch,model):
  """
  """
  model.eval()
  with torch.no_grad():
    batch = (item.to(device) for item in batch)
    input_ids, token_type_ids, mc_token_ids, lm_labels, mc_labels = batch
    outputs = model(input_ids = input_ids, mc_token_ids = mc_token_ids, mc_labels = mc_labels,
                  lm_labels = lm_labels, token_type_ids = token_type_ids)
    lm_loss, mc_loss = outputs[0], outputs[1]
    total_loss = lm_loss * args.lm_coef + mc_loss * args.mc_coef
  return lm_loss.item(),mc_loss.item(),total_loss.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPT2 summarizer')
    parser.add_argument('--epochs', type=int, default=1, metavar='E',
                        help='number of epochs to train for (default: 5)')

    parser.add_argument('--learning_rate', type=float, default=5e-5, metavar='de_LR',
                        help='learning rate of the model (default: 5e-5)')
    parser.add_argument('--eps', type=float, default=1e-8, metavar='de_LR',
                        help='epsilon for the AdamW to prevent dividing by 0 (default: 1e-8)')
    parser.add_argument('--max_norm', type=float, default=1.0,
                        help='gradient norm clipping (default:1.0)')
    parser.add_argument('--lm_coef', type=float, default=2.0,
                        help='weight of the lm loss (default:1.0)')
    parser.add_argument('--mc_coef', type=float, default=1.0,
                        help='weight of the mc loss (default:1.0)')
    parser.add_argument('--grad_accumulation', type=int, default=5,
                        help='gradient accumulation (since batch size not implemented) (default:5)')
    parser.add_argument('--scheduler_warmup', type=int, default=50,
                        help='steps of iterations for learning rate scheduler warm up (default:50)')
    parser.add_argument('--print_every', type=int, default=100,
                        help='steps of iterations before printing the loss (default:100)')

    parser.add_argument('--train_data', type=str,help='path to train tensor data .pt')
    parser.add_argument('--val_data', type=str,help='path to validation tensor data .pt')
    parser.add_argument('--model_name', type=str, default='GPT2_folder',
                        help='name of the model directory to be saved in (default: GPT2_folder)')

    parser.add_argument('--model_directory', type=str, default = None,
                        help='path to the GPT2 model directory (must contain pytorch_model.bin, config.json, vocab.json, and merges.txt)')
    main(parser.parse_args())
