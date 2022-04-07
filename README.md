# MLE: Containerization

## Dataset

For this competition, you will be predicting a binary target based on a number of feature columns given in the data. The columns are a mix of scaled continuous features and binary features.


The data is synthetically generated by a GAN that was trained on real-world molecular response data.


Data contains 286 features and 1kk rows but we will use only 100k for training.


[dataset link](https://www.kaggle.com/c/tabular-playground-series-oct-2021/data)


## Model
We will use Linear regression from sklearn to predict target variable.

## Build container

`docker build -t logreg .`

## Run container

`docker-compose up -d && docker attach logreg_cont`


To see how it works:

- Put the `train.csv` file with training data to `volume/`
- (In terminal) train the model: `python src/train.py volume/train.csv`
- You will see `/volume/logreg.pkl` and `volume/scaler.pkl` - the result artifacts after training model.