---
title: Sentance Transmorgrifier
emoji: s
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 3.8.2
app_file: app.py
pinned: false
license: apache-2.0
---

## Sentance Transmorgrifier

# What is the Sentance Transmorgrifier?
- The Sentance Transmorgrifier is a generic text to text conversion tool which uses a categorical gradiant boost library, [catboost](https://catboost.ai/), as its back end.
- This library does not use neural net or word embeddings but does the transformation on the character level.
- For Sentance Transmorgrifier to work, there has to be some common characters between the from and two conversion.
- The model uses a modified form of the [logest common subsequence algorithm](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem) to transform the sentance conversion into a sequence of three types of operations:
  1. Match: Pass the character from input to output
  2. Drop: Remove the incoming character from the input.
  3. Insert: Generate a character and add it to the output.
- The transformation uses a sliding context window of the next n incoming characters, n output transformed chars and n output untransformed chars.
- Because the window is sliding, there is no fixed length on the character sequences which can be transformed.

# How can I use the Sentance Transmorgrifier 
- The project has been configured to be able to be used in two different ways.

## Shell access
- The transmorgrify.py script can be called directly with arguments specifying an input csv file, what lables are from and to and what to save the resulting model as or to process the input csv to an output.  Here is an example:

```sh
python transmorgrify.py \
    --train --in_csv /home/lansford/Sync/projects/tf_over/sentance_transmogrifier/examples/phonetic/phonetic.csv \
     --a_header English \
     --b_header Phonetic\
     --device 0:1 \
     --model phonetics_gpu_4000.tm \
     --verbose \
     --iterations 4000 \
     --train_percentage 50
```
 - `--train` This says that the system is supposed to train as apposed to doing inference.
 - `--a_header` This indicates the header in the csv file which identifies the from column
 - `--b_header` This indicates the to column
 - `--device` This specifies the gpu if you have one or type `cpu` if you do not have a gpu.
 - `--model` This indicates where to save the model
 - `--verbose` Self explanitory
 - `--iterations` This indicates how many catboost iterations should be executed on your input data.
 - `--train_percentage` If you are going to use the same file for testing as well as the training, giving a train percentage will only use the percentage specified for training.

```sh
python transmorgrify.py \
    --execute \
    --in_csv /home/lansford/Sync/projects/tf_over/sentance_transmogrifier/examples/phonetic/phonetic.csv \
    --a_header English \
    --b_header Phonetic\
    --device cpu \
    --model phonetics_gpu_4000.tm \
    --verbose \
    --include_stats \
    --out_csv ./phonetics_out_gpu_4000.csv \
    --train_percentage 50
```
 - `--execute` This indicates that this is supposed to execute the model as apposed to training it.
 - `--a_header` This indicates the header in the csv file which identifies the from column.
 - `--b_header` This indicates the to column.  The to column must be specified if `--include_stats` is also specified.
 - `--device` This specifies the gpu if you have one or type `cpu` if you do not have a gpu.
 - `--model` This indicates where to load the model
 - `--verbose` Self explanitory
 - `--include_stats` This adds editing distance to the output csv so that you can sort and graph how well the model did.  It reports the Levenshtein Distance from input to output before and after transformation and the percent improvement.
 - `--out_csv` This indicates where the data should be saved after being processed by the model.
 - `--train_percentage` If you are going to use the same file for testing as well as the training, give the same train percentage as was given for training and the execution will only use the remaining data not used for training.

 ## Python object access
 - If instead of wanting to run this from the command line you want to use this in a python app, you can import the Transmorgrifier from the transmorgrify.py file into your app and use the methods on it.  This model has the following functions.
  - `train`
  ```
Train the Transmorgrifier model.  This does not save it to disk but just trains in memory.

Keyword arguments:
from_sentances -- An array of strings for the input sentances.
to_sentances -- An array of strings of the same length as from_sentances which the model is to train to convert to.
iterations -- An integer specifying the number of iterations to convert from or to. (default 4000)
device -- The gpu reference which catboost wants or "cpu". (default cpu)
trailing_context -- The number of characters after the action point to include for context. (default 7)
leading_context -- The number of characters before the action point to include for context. (default 7)
verbose -- Increased the amount of text output during training. (default True)
  ```
  - `save`
```
Saves the model previously trained with train to a specified model file.

Keyword arguments:
model -- The pathname to save the model such as "my_model.tm"
```
  - `load`
```
Loads the model previously saved from the file system.

Keyword arguments:
model -- The filename of the model to load. (default my_model.tm)
```
  - `execute`
```
Runs the data from from_sentaces.  The results are returned 
using yield so you need to wrap this in list() if you want 
to index it.  from_sentances can be an array or a generator.

Keyword arguments:
from_sentances -- Something iterable which returns strings.
```
- Here is an example of using object access to train a model
```python
import pandas as pd
import transmorgrify

#load training data
train_data = pd.read_csv( "training.csv" )

#do the training
my_model = transmorgrify.Transmorgrifier()
my_model.train( 
    from_sentances=train_data["from_header"], 
    to_sentances=train_data["to_header"],
    iterations=4000 )

#save the results
my_model.save( "my_model.tm" )
```

- Here is an example of using object access to use a model
```python
import pandas as pd
import transmorgrify

#load inference data
inference_data = pd.read_csv( "inference.csv" )

#Load the model
my_model = transmorgrify.Transmorgrifier()
my_model.load( "my_model.tm" )

#do the inference
#model returns a generator so wrap it with a list
results = list( my_model.execute( inference_data["from_header"] ) )
```
