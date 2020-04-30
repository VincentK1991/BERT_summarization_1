# BERT_summarization_1
 
Tutorial for beginners, first time BERT users. Also a text summarization tool, useing BERT encoder, and topic clustering approach. This is what's called "extractive summarization", meaning, a key sentences containing crucial information is extracted from the paragraph.

# Furthermore, GPT-2
As a companion to BERT, I added GPT2 summarization. This is so-called "abstractive summarization". This part is very much work in progress, since I still haven't figured out a lot of the training. 

# update on GPT2 training
I figured out how to train GPT2 model to a good outcome. The training is shown in the Ignite Engine notebook. The outcome of this is shown in the example notebook on BERTandGPT2_abstractive_summarization_Apr28_2020.

# Using Ignite Engine for training GPT2
Ignite is a pytorch-based library to help with training neural networks. We use this to write a compack code for training GPT2. 
