import numpy as np
from utils.preprocessing import load_and_preprocess
from models.lstm_model import build_lstm

X_train, X_test, y_train, y_test, scaler = \
    load_and_preprocess("dataset/flood_data.csv")

X_train = X_train.reshape(X_train.shape[0],
                          X_train.shape[1], 1)

X_test = X_test.reshape(X_test.shape[0],
                        X_test.shape[1], 1)

model = build_lstm((X_train.shape[1],1))

model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

model.save("flood_model.h5")

