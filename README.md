# The pre-print article is out! 
please visit, and suggest if you want to see any changes. I thanks our co-authors/collaborators Bowen Tan and Yiming Niu from Rockefeller University. 
https://arxiv.org/abs/2006.01997

---

# command line interface

I added a more user friendly command line pre-processing/training/summarization codes for the GPT2. These are the GPT2_preprocessing.py, trainGPT2.py, and GPT2_summarizer.py. To use it, first you'd need Huggingface's transformer package, and a folder where you'd want to save your fine-tuned model on.
For the training and validation dataset, refer to the notebook *pre-processing-text-for-GPT2-fine-tuning*.
(Update on Aug 21 2020)

## setting up the environment
To install from the Pipfile

`pipenv install`

or to install from the requirements.txt

`pip install -r requirements.txt`

## pre-processing

The code is in the GPT2_preprocessing.py and the helperGPT2.py.
The pre-processing takes the input = metadata.csv given in the [COVID19 Open Research dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge)

`python GPT2_preprocessing.py --input=metadata.csv`

the output is the pytorch tensordataset

## Training the GPT2

Use the pytorch tensordataset to train the GPT2 (preferrably you separate the training and validation dataset how you like it)

`mkdir fine_tuned_folder`

`python train_command_line.py --epochs=1 --train_data='insert-your-training-data-here' --val_data='insert-your-validation-data-here' --model_name='fine_tuned_folder'`

You'd need GPU and cuda to train GPT2. 100 iterations took me about 44 seconds on 1 Nvidia Tesla P-100.

## Generating the summary

To generate a summary give the input file as .txt file and the model directory where the pytorch_model.bin and the config files are kept.

`python GPT2_summarize.py --input_file='input.txt' --model_directory='insert-your-model-directory-here'`

---

# Directory

## notebook

This folder contains colab notebooks that guide you through the summarization by BERT and GPT-2. You should be able to open it on your Google's colab, and play with your data. *The text .csv file, the post-processed training tensor file, and fine-tuned model weight tensor are available upon request.* 

Both BERT and GPT-2 models are implemented in the Transformer library by Huggingface. The description of each notebooks are listed below. The citation and related works are in the "generate-summary-with-BERT-or-GPT2" notebook.

### Primer-to-BERT-extractive-summarization
 
Tutorial for beginners, first time BERT users. Also a text summarization tool, useing BERT encoder, and topic clustering approach. This is what's called "extractive summarization", meaning, a key sentences containing crucial information is extracted from the paragraph.

As a companion to BERT, I added GPT2 summarization. This is so-called "abstractive summarization". I fine-tune the already pre-trained GPT2 for specific summarization task.

### pre-processing-text-for-GPT2-fine-tuning
This notebook guides you through a pre-processing that turn the text data to a tokenized tensors that is ready for the training. The raw data can be obtained from this [website](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).

### fine-tune-GPT2-for-summarization
I use Ignite, which is a pytorch-based library to help keeping track of training. The data comes from the pre-processing step in the previous notebook.

### generate-summary-with-BERT-or-GPT2
I figured out how to train GPT2 model to a reasonable outcome. This notebook summarizes how the data is processed from the text format to a tokenized query for summarization.
