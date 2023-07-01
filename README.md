# <center> Stock Adjusted Close Price Prediction Using RWKV </center>

Hi! In this project we used RWKV architecture - a novel architecture trying to push RNNs to the transformers era, modified it to work with stock numerical values and trained it to predict future values. We've extracted features of the stock of The Coca-Cola Company (KO) and trained the data with those features. 

<p align="center">
Watch on Youtube:  <a href=""><img src="https://img.shields.io/badge/-YouTube-red?&style=for-the-badge&logo=youtube&logoColor=white" height=20></a>
</p>
 
- [Stock Adjusted Close Price Prediction Using RWKV](#Stock-Adjusted-Close-Price-Prediction-Using-RWKV)
  * [Previous Work](#Previous-Work)
  * [Data Processing](#Data-Processing)
  * [Architecture](#Architecture)
  * [Hyperparameters](#Hyperparameters)
  * [Result](#Result)
  * [Usage](#Usage)
  * [Files in the Repository](#Files-in-the-Repository)
  * [Further Work](#Further-Work)


## Previous Work
Bitcoin prediction using RNN:

https://www.kaggle.com/muharremyasar/btc-historical-with-rnn

IBM stock price prediction using Transformer-Encoder:

https://github.com/JanSchm/CapMarket/blob/master/bot_experiments/IBM_Transformer%2BTimeEmbedding.ipynb

Short term stock price prediction using LSTM with a simple trading bot:

https://github.com/roeeben/Stock-Price-Prediction-With-a-Bot/blob/main/README.md

Bitcoin Price Prediction Using Transformers:

https://github.com/baruch1192/-Bitcoin-Price-Prediction-Using-Transformers

Predicting Stock Prices with Deep Neural Networks (LSTM)

https://www.alphavantage.co/academy/#lstm-for-finance

## Data Processing
We are using The Coca-Cola company historical daily stock data, from: 04/12/1981 till 29/06/2023 using 'yfinance' package.
The data contains 6 features: Open price, Close price, High price, Low price, Volume, and Adjusted close price. 
Since our model didn't handle well the raw data, we've decided to apply moving average with a window size of 10. 
All of the data was scaled together using MinMax scaler. 

After that we split the data into train (80%), validation (10%), and test (10%), in chronological order as can be seen here:

![alt text](https://github.com/tomer9080/Stock-Prediction-Using-RWKV/blob/main/images/data_set_split.png)

Finally we've divided the train set to windows, where each window is comprised of `window_size` samples of the features, and combined few windows together to create a batch for train, validation and test. 

## Architecture
We used PyTorch HuggingFace's RWKV model. The original model was compromised of embedding layer, which we decided to remove, 
since we wanted the NN to handle numerical data. The RWKV model is implemented as in the pubilshed article: [RWKV](https://arxiv.org/pdf/2305.13048.pdf).


The model structure, including our added layers:
<p align="center">
  <img src="https://github.com/tomer9080/Stock-Prediction-Using-RWKV/blob/main/images/RWKV_ARCH.png" width="450"/>
</p>


## Hyperparameters
* `batch_size` = int, size of batch
* `epochs` = int, number of epochs to run the training
* `window_size` = int, the length of the sequence
* `hidden_states` = int, the number of hidden states in eahc RWKV block
* `hidden_layers` = int, the number of RWKV blocks in the NN.
* `dropout` = float, the dropout probability of the dropout layers in the model (0.0 - 1.0)
* `lr` = float, starting learning rate 
* `factor` = float, multiplicative factor of learning rate decay (0.0 - 1.0)
* `patience` = int, how many epochs we'll wait before decaying lr after no improvement

The most crucial thing to understand here is the relations between `bptt_src`, `bptt_tgt` and `overlap`. We use `bptt_src` past samples to predict the following `bptt_tgt - overlap`.


## Result

We trained the model with the hyperparameters:

|Param|Value|
|-------|-----|
|`window_size` | 40 |
|`hidden_layers`| 8 |
|`hidden_states`| 32 |
|`dropout`| 0.2 |
|`epochs`| 30 |
|`batch_size`| 128 |
|`lr`| 0.01 |
|`factor`| 0.6 |
|`patience`| 1 |

And we got the results:

<p align="center">
  <img src="https://github.com/tomer9080/Stock-Prediction-Using-RWKV/blob/main/images/predictions_all.png" />
</p>

We can see that the trend if the predicted values, is similar to the original trend, and even that in the train and validation areas, we are giving pretty accurate prediciton.

Let's have a zoom in to the test prediction:

<p align="center">
  <img src="https://github.com/tomer9080/Stock-Prediction-Using-RWKV/blob/main/images/predictions_test.png" />
</p>

Here we can see that the predicted trend behaves well, but after sometime it seems that we are losing resolution and diverging from the real stock values, although we are having success identifying sharp movements. We can also see how the fact that we've used moving average has smoothened our prediction, and it easy to observe how less spiky it is.

## Usage

To retrain the model run [stock_prediction_using_rwkv.ipynb](https://github.com/tomer9080/Stock-Prediction-Using-RWKV/stock_prediction_using_rwkv.ipynb). You can choose different stock to predict on in the relvant cell by just riplacing the ticker, and deciding on how much days you want to train (notice that different stocks has different number of data points). after you chose your hyperparameters, run all of the notebook and wait untill it's done.


## Files in the repository

| Folder |File name         | Purpose |
|------|----------------------|------|
|code|`stock_prediction_using_rwkv.ipynb`| Notebook which includes all data processing, training, and inference |
|images|`RWKV_ARCH.png`| Image that shows our arch including the RWKV model |
| |`data_set_split.png`| Image that shows our data split |
| |`predictions_all.png`| Image that shows the predictions obtained on all sets |
| |`predictions_test.png`| Image that shows our result on the test set |


## Further Work

The work we presented here achieved good results, but definitely there are aspects to improve and examine such as:
- Try running the model on a different stock.
- Examine adding Time2Vec embedding.
- Try and train the model on multiple stocks, and predict on them.

Hope this was helpful and please let us know if you have any comments on this work:

https://github.com/tomer9080

https://github.com/roilatzres