import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

plt.style.use("fivethirtyeight")


def lstm_price_prediction(company, start=dt.date(2012, 1, 1), end=dt.date.today()):

    # Load stock data
    data = web.DataReader(company, "yahoo", start, end)

    # Prepare data
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

    # Build the model
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
    # Load test data
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

    # Make predictions on test data
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


def svr_price_prediction(company, start=dt.date(2022, 3, 1), end=dt.date.today()):

    # Load stock data
    df = web.DataReader(company, "yahoo", start, end)

    # Store the last row of data
    actual_price = df.tail(1)
    # Get all of the data except the last row
    df = df.head(len(df) - 1)

    # Create empty lists
    days = list()
    close_prices = list()

    # Get only the dates and the close prices
    df_days = df.index.date
    df_price = df.loc[:, "Close"]

    # Create the independent data set (dates)
    for day in df_days:
        days.append([int(str(day).split("-")[2])])
    # Create the dependent data set (close prices)
    for close_price in df_price:
        close_prices.append(float(close_price))

    # Create and train an SVR model using a linear kernel
    lin_svr = SVR(kernel="linear", C=1000.0)
    lin_svr.fit(days, close_prices)
    # Create and train an SVR model using a polynomial kernel
    poly_svr = SVR(kernel="poly", C=1000.0, degree=2)
    poly_svr.fit(days, close_prices)
    # Create and train an SVR model using a RBF kernel
    rbf_svr = SVR(kernel="rbf", C=1000.0, gamma=0.85)
    rbf_svr.fit(days, close_prices)

    # Plot the models on a graph to see which has the best fit
    plt.figure(figsize=(16, 8))
    plt.scatter(days, close_prices, color="black", label="Original Data")
    plt.plot(days, lin_svr.predict(days), color="blue", label="Linear Model")
    plt.plot(days, poly_svr.predict(days), color="orange", label="Polynomial Model")
    plt.plot(days, rbf_svr.predict(days), color="green", label="RBF Model")
    plt.xlabel("Days")
    plt.ylabel("Close Price ($)")
    plt.title(f"{company} Price Prediction ({start} - {end})")
    plt.legend()
    plt.savefig(f"./plot/{company}_SVR_predicted.png")

    # Show the predicted price for the given day
    day = [[str(end.day)]]
    print(f"The close price for the day {end}.")
    print("The Linear SVR predicted:", lin_svr.predict(day))
    print("The Polynomial SVR predicted:", poly_svr.predict(day))
    print("The RBF SVR predicted:", rbf_svr.predict(day))

    
    # Show the actual price
    print("The actual price:", actual_price['Close'][0])
    
