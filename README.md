# Directory

## notebook

This folder contains colab notebooks that guide you through the summarization by BERT and GPT-2. You should be able to open it on your Google's colab, and play with your data. *The text .csv file, the post-processed training tensor file, and fine-tuned model weight tensor are available upon request.* 

Both BERT and GPT-2 models are implemented in the Transformer library by HUggingface. The description of each notebooks are listed below. The citation and related works are in the "generate-summary-with-BERT-or-GPT2" notebook.

### Primer-to-BERT-extractive-summarization
 
Tutorial for beginners, first time BERT users. Also a text summarization tool, useing BERT encoder, and topic clustering approach. This is what's called "extractive summarization", meaning, a key sentences containing crucial information is extracted from the paragraph.

As a companion to BERT, I added GPT2 summarization. This is so-called "abstractive summarization". I fine-tune the already pre-trained GPT2 for specific summarization task.

### pre-processing-text-for-GPT2-fine-tuning
This notebook guides you through a pre-processing that turn the text data to a tokenized tensors that is ready for the training. The raw data can be obtained from this [website](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge).

### fine-tune-GPT2-for-summarization
I use Ignite, which is a pytorch-based library to help keeping track of training. The data comes from the pre-processing step in the previous notebook.

### generate-summary-with-BERT-or-GPT2
I figured out how to train GPT2 model to a reasonable outcome. This notebook summarizes how the data is processed from the text format to a tokenized query for summarization.
