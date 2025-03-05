# Tokenizers

Tokenizers are the first stage in the "LLM Pipeline." They convert text into "tokens" (a unit that can be processed by a transformer).
This repository is a template for getting started with your own tokenizer, and contains method stubs with documentation on how to implement
the methods. 

# Getting Started

## Prerequisites

- Python 3.8, 3.9, 3.10, 3.11, 3.12
  - https://www.python.org/downloads/
- Shakespeare's texts
  - You can use any text you want, but Shakespeare's texts are included in `./data/shakespeare.txt`

## Installation
1. Fork the repository
2. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/tokenizers.git
    ```
3. Open `tokenizer.py`

You you will need to the regex package (python's built in `re` will not be good enough). You can install it with `pip`:
```bash
pip install regex
# or
pip3 install regex
```

There aren't any other libraries we need to install today, but for actual transformer training, you will
probably want `torch` and maybe a prebuilt tokenizer like `huggingface/tokenizers`.



# What is a Tokenizer?

A tokenizer is a tool that consistently converts text into tokens, and vice versa. The argument is that the meaning of a sentence doesn't 
necessarily come from the letters, but from the words. So, we can compress the entire text by merging letters together into words. This way,
we can losslessly compress the text into a smaller representation. 

By doing this compression, we increase our context window (the amount of data we can put into the LLM at one time). 

Generally, tokenizers are considered a necessary evil. The compression leads to annoying artifacts which makes training a bit harder, makes
inference have limitations in terms of spelling and letters, and makes multi-langual inputs less optimal. Researchers are working on ways
to use transformers without tokenizers.


