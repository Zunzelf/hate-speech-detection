# Hate-Speech-Detection
This repository used for storing task to fullfill NLP college project, which is identifying hate speech from text(mostly from tweet)

## Data
The data we use in this repository is from [t - davidson](https://github.com/t-davidson/hate-speech-and-offensive-language/tree/master/data) with formatted such as  :
- `class` : 0 - hate speech, 1 - offensive language, 2 - neither
- `tweet` : tweet/text


## How-to-use
To use this repository, you need to have a word vector model, then you may generate the dataset using that vector model (we like to call it token-vectorized) after that you will able to train our classifier model using the dataset.

### word vector
In this project, vector model that we use is **word2vec** model from gensim, we create the model from the dataset's texts (yea, we know it's not good, but well it's just to speed thigs up), but feel free to use your own vector model, and put it in "utils" folder.

### Generate-Dataset
To generate the dataset you need to run `dataset.py` by using :
> cd **utils**
>
> py **<span>dataset.py</span>**

Then the data will be ready in the `utils` folder (and as we said earlier, we like to call it token-vectorized thus we named it the same~).

### Train-Model
First of all, the model we use in this project is one of the Deep-Learning variants called LSTM(Long-Short-Term Memory), and to train the model you may use the `app.py` by :
> py **<span>app.py</span>**

### Predicts
In this project, we didn't implement predict function yet (hopefully soon)
