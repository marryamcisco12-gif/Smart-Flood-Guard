import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess(path):

    data = pd.read_csv(path)
    data = data.dropna()

    features = ['Rainfall', 'River_Level', 'Temperature', 'Humidity']
    target = 'Flood_Risk'

    X = data[features]
    y = data[target]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, scaler
  
