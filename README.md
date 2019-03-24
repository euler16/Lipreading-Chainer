# Lipreading [Chainer](chainer_idon_red.png)
<center>[](chainer_idon_red.png)</center>
This is the chainer code for the paper *Combining Residual Networks with LSTMs for Lipeading*. You can find the paper [here](https://arxiv.org/pdf/1703.04105.pdf).<br>
The authors present a word level lipreading model based on Resnets. The input to the model is a silent video and the model then outputs the word it thinks was spoken. In the paper this task of visual speech recognition has been modelled as video classification.

The code is based on PyTorch implementation of the same work which can be found [here](https://github.com/mpc001/end-to-end-Lipreading).
## Dataset 
The model has been trained on Oxford-BBC *Lip Reading in the Wild (LRW)* dataset. The dataset consists of short video clips of news anchor speaking a single word. The words dictionary size is 500. The dataset contains about 1000 utterances of each of the 500 words. Dataset size is around 70GB.
## How to Run
1. Download the LRW dataset from [this website](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)
2. Preprocess the dataset as given in the PyTorch counterpart of this repo (available [here](https://gist.github.com/shaform/7cdba07f2bb17a9f72697253732a1f1c))
3. Write the appropriate dataset path in config.json
4. Run the following command :-
```
python main.py --config config.json
```
5. after the training is over, change the mode variable in config.json to 'backendGRU' and run the above command.
6. Finally fine tune the model by switching the mode to 'finetuneGRU'.

Make sure you change the path variable to saved model location after step 4.
## TODOs
- [x] Chainer code, tested
- [x] Tested on CPU
- [ ] Making it work on GPU 
