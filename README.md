# StackOverflow Question Tagging Toolkit
This toolkit provides a StacOverflow question embedding using LDA/BERT/LDA+BERT and an DNN based tags prediction.

## Introduction
This project intended to predict tags to StackOverflow question. It contains text embedding using 3 methods as suggested in:
URL: https://blog.insightdatascience.com/contextual-topic-identification-4291d256a032

The methods are:
1) LDA
2) BERT
3) LDA and BERT

The Tags prediction is carried using a DNN. 

## DNN architecture:
3 hidden layesr, 256 units each with relu activation. Output layer with 100 units and sigmoid activation function. For more info, see stage 3 in main.py.

## Data:
The data used in this project is the StackSample: 10% of Stack Overflow Q&A.
URL: https://www.kaggle.com/stackoverflow/stacksample

## Usage:
The main.py file contains 4 stages that can be run individually:
Stage 0: Data preparation, which includes filtering the data to contain only the 100 most used tags in the database
Stage 1: Question embedding and tags binarization
Stage 2: Data visualization
Stage 3: Classification


### main.py script options:
m 0 : LDA
m 1 : BERT
m 2 : LDA+BERT
s : starting stage (0), (1), (2) or (3)
e : exiting stage (0), (1), (2) or (3)
To run the entire program set s to 0 and e to 3

### Run:
python3 $main -m 2 -s 0 -e 3 --prj_dir=path/to/dir

## License
MIT License

Copyright (c) [2020] [Meital Rannon]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
