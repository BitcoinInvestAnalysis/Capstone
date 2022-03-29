import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


def lstm_price_prediction(company, start=dt.date(2012, 1, 1), end=dt.date.today()):

    # Load Data
    data = web.DataReader(company, "yahoo", start, end)

    # Prepare Data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))

    prediction_days = 60
    future_day = 30

    X_train, y_train = [], []

    for x in range(prediction_days, len(scaled_data) - future_day):
        X_train.append(scaled_data[x - prediction_days : x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build the Model
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))  # prevent overfitting
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # prediction of the next closing value

    model.compile(optimizer="adam", loss="mean_squared_error")
    model.fit(X_train, y_train, epochs=25, batch_size=32)

    # Test the model accuracy on existing data
    # Load Test Data
    test_start = dt.date(2020, 1, 1) + dt.timedelta(days=-prediction_days)
    test_end = dt.date.today()

    test_data = web.DataReader(company, "yahoo", test_start, test_end)
    actual_prices = test_data["Close"].values

    total_dataset = pd.concat((data["Close"], test_data["Close"]), axis=0)

    model_inputs = total_dataset[
        len(total_dataset) - len(test_data) - prediction_days :
    ].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    # Make Predictions on Test Data
    X_test = []

    for x in range(prediction_days, len(model_inputs)):
        X_test.append(model_inputs[x - prediction_days : x, 0])

    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Plot the test predictions
    plt.figure(figsize=(16, 8))
    plt.plot(actual_prices, color="black", label=f"Actual {company} Price")
    plt.plot(predicted_prices, color="green", label=f"Predicted {company} Price")
    plt.title(f"{company} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f"{company} Share Price")
    plt.legend(loc="upper left")
    plt.savefig(f"./plot/{company}_LSTM_predicted.png")
