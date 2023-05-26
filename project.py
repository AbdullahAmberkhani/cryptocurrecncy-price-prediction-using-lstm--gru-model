import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU, LSTM
from keras import optimizers
import tkinter as tk
import yfinance as yf
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

root = tk.Tk()
root.title("Cryptocurrency Price Prediction")

# Create labels
ticker_label = tk.Label(root, text="Enter the cryptocurrency:")
ticker_label.pack()

# Create dropdown menu
crypto_list = ['BTC-USD', 'ETH-USD', 'USDT-USD', 'BNB-USD', 'USDC-USD', 'XRP-USD', 'ADA-USD', 'LUNA-USD', 'SOL-USD', 'AVAX-USD']
selected_crypto = tk.StringVar()
selected_crypto.set(crypto_list[0])
crypto_dropdown = tk.OptionMenu(root, selected_crypto, *crypto_list)
crypto_dropdown.pack()
 
# Create loading screen
loading_screen = tk.Toplevel()
loading_screen.title("Loading...")
loading_label = tk.Label(loading_screen, text="Please wait while the prediction is being processed...")
loading_label.pack()

# Define function to predict price
def predict_price():
        # Reset the app by clearing the text and removing the existing graphs
    message_text.delete(1.0, tk.END)
    if 'canvas1' in locals():
        canvas1.get_tk_widget().pack_forget()
    if 'canvas2' in locals():
        canvas2.get_tk_widget().pack_forget()
    if 'canvas3' in locals():
        canvas3.get_tk_widget().pack_forget()

    loading_screen.deiconify()

    seed = 1234
    np.random.seed(seed)
    plt.style.use('ggplot')

    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    ticker = selected_crypto.get()

    # Download data from Yahoo Finance
    dataraw =  yf.download(ticker)

    # use feature 'Date' & 'Close'
    dataset = pd.DataFrame(dataraw['Close'])
    
    #Min-Max Normalization
    dataset_norm = dataset.copy()
    dataset[['Close']]
    scaler = MinMaxScaler()
    dataset_norm['Close'] = scaler.fit_transform(dataset[['Close']])
    
    # Partition data into data train, val & test
    totaldata = dataset.values
    totaldatatrain = int(len(totaldata)*0.7)
    totaldataval = int(len(totaldata)*0.1)
    totaldatatest = int(len(totaldata)*0.2)

    # Store data into each partition
    training_set = dataset_norm[0:totaldatatrain]
    val_set=dataset_norm[totaldatatrain:totaldatatrain+totaldataval]
    test_set = dataset_norm[totaldatatrain+totaldataval:]

    # Initiaton value of lag
    lag = 2
    # sliding windows function
    def create_sliding_windows(data,len_data,lag):
        x=[]
        y=[]
        for i in range(lag,len_data):
            x.append(data[i-lag:i,0])
            y.append(data[i,0]) 
        return np.array(x),np.array(y)

    # Formating data into array for create sliding windows
    array_training_set = np.array(training_set)
    array_val_set = np.array(val_set)
    array_test_set = np.array(test_set)

    # Create sliding windows into training data
    x_train, y_train = create_sliding_windows(array_training_set,len(array_training_set), lag)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
    # Create sliding windows into validation data
    x_val,y_val = create_sliding_windows(array_val_set,len(array_val_set),lag)
    x_val = np.reshape(x_val, (x_val.shape[0],x_val.shape[1],1))
    # Create sliding windows into test data
    x_test,y_test = create_sliding_windows(array_test_set,len(array_test_set),lag)
    x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

    # Hyperparameters
    learning_rate = 0.0001
    hidden_unit = 64
    batch_size = 256
    epochs = 100

    # Define GRU model
    regressorGRU = Sequential()
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=True, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    regressorGRU.add(GRU(units=hidden_unit, return_sequences=False, activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    regressorGRU.add(Dense(units=1))
    regressorGRU.compile(optimizer=optimizers.Adam(learning_rate), loss='mean_squared_error')

    # Define LSTM model
    regressorLSTM = Sequential()
    regressorLSTM.add(LSTM(units=hidden_unit, return_sequences=True, input_shape=(x_train.shape[1],1), activation='tanh'))
    regressorLSTM.add(Dropout(0.2))
    regressorLSTM.add(LSTM(units=hidden_unit, return_sequences=True, activation='tanh'))
    regressorLSTM.add(Dropout(0.2))
    regressorLSTM.add(LSTM(units=hidden_unit, return_sequences=False, activation='tanh'))
    regressorLSTM.add(Dropout(0.2))
    regressorLSTM.add(Dense(units=1))
    regressorLSTM.compile(optimizer=optimizers.Adam(learning_rate), loss='mean_squared_error')


    # Train the models
    history_gru = regressorGRU.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs)
    history_lstm = regressorLSTM.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs)
    # Combine predictions from both models
    pred_gru = regressorGRU.predict(x_test)
    pred_lstm = regressorLSTM.predict(x_test)
    pred = (pred_gru + pred_lstm) / 2

    # Get the train and validation loss from the history object
    train_loss_gru = history_gru.history['loss'][-1]
    val_loss_gru = history_gru.history['val_loss'][-1]
    train_loss_lstm = history_lstm.history['loss'][-1]
    val_loss_lstm = history_lstm.history['val_loss'][-1]

    # Print the train and validation loss
    print(f'GRU model train loss: {train_loss_gru:.4f}')
    print(f'GRU model validation loss: {val_loss_gru:.4f}')
    print(f'LSTM model train loss: {train_loss_lstm:.4f}')
    print(f'LSTM model validation loss: {val_loss_lstm:.4f}')

    # Create a figure and axis for the second graph (Price Prediction)
    figure2 = plt.Figure(figsize=(6, 4), dpi=100)
    ax2 = figure2.add_subplot(111)
    ax2.plot(y_test, color='red', label='Actual Price')
    ax2.plot(pred, color='blue', label='Predicted Price')
    ax2.set_title('Price Prediction')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.legend()

    # Create a canvas for the second graph
    canvas2 = FigureCanvasTkAgg(figure2, master=root)
    canvas2.draw()
    canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)



    # Tabel value of training loss & validation loss
    # Create a dataframe to hold the loss values
    loss_df = pd.DataFrame({
        'Model': ['GRU', 'LSTM'],
        'Train Loss': [train_loss_gru, train_loss_lstm],
        'Val Loss': [val_loss_gru, val_loss_lstm]
    })

    # Print the dataframe
    print(loss_df.to_string(index=False))

    # Implementation model into data test
    y_pred = (regressorGRU.predict(x_test)+regressorLSTM.predict(x_test))/2

    # Rescale the predicted and actual values
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Calculate the RMSE
    mse = np.mean((y_test - y_pred)**2)
    rmse = np.sqrt(mse)
    #message_text.insert(tk.END, "The Root Mean Squared Error is:" + str(rmse) + "\n")

    # Calculate the MAPE
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    message_text.insert(tk.END, "The Mean Absolute Percentage Error is: " + str(mape) + "\n")

    # Comparison data test with data prediction
    datacompare = pd.DataFrame()
    datatest=np.array(dataset['Close'][totaldatatrain+totaldataval+lag:])
    datapred= y_pred

    datacompare['Data Test'] = datatest
    datacompare['Prediction Results'] = datapred
    datacompare

    last_data = dataset[-lag:].values
    last_data_norm = scaler.transform(last_data)
    last_data_norm = last_data_norm.reshape(1, lag, 1)

    # Predict next day's price using the combined model
    predicted_price = (regressorGRU.predict(last_data_norm) + regressorLSTM.predict(last_data_norm)) / 2

    # Inverse transform the predicted price to get the actual price
    predicted_price = scaler.inverse_transform(predicted_price)[0][0]


    # Determine if the price will increase or decrease
    current_price = dataraw['Close'][-1]
    next_day_price = predicted_price
    if next_day_price > current_price:
        message_text.insert(tk.END, "The Price is predicted to Increase Tomorrow." + "\n")
    else:
        message_text.insert(tk.END, "The Price is predicted to Decrease Tomorrow." + "\n")

    message_text.insert(tk.END, "Todays Price is " + str(current_price) + "\n")
    message_text.insert(tk.END, "Tomorrows price is predicted to be " + str(predicted_price) + "\n")


    # Create a figure and axis for the first graph (GRU vs LSTM Model Loss)
    figure1 = plt.Figure(figsize=(6, 4), dpi=100)
    ax1 = figure1.add_subplot(111)
    ax1.plot(history_gru.history['loss'], label='train_loss_gru')
    ax1.plot(history_gru.history['val_loss'], label='val_loss_gru')
    ax1.plot(history_lstm.history['loss'], label='train_loss_lstm')
    ax1.plot(history_lstm.history['val_loss'], label='val_loss_lstm')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('GRU vs LSTM Model Loss')
    ax1.legend()

    # Create a canvas for the first graph
    canvas1 = FigureCanvasTkAgg(figure1, master=root)
    canvas1.draw()
    canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Create a figure and axis for the third graph (Combined Loss)
    figure3 = plt.Figure(figsize=(8, 6), dpi=100)
    ax3 = figure3.add_subplot(111)
    combined_loss = (np.array(history_gru.history['loss']) + np.array(history_lstm.history['loss'])) / 2
    combined_val_loss = (np.array(history_gru.history['val_loss']) + np.array(history_lstm.history['val_loss'])) / 2
    ax3.plot(combined_loss, label='Combined Train Loss')
    ax3.plot(combined_val_loss, label='Combined Validation Loss')
    ax3.set_title('Combined GRU-LSTM Model Loss')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Loss')
    ax3.legend()

    # Create a canvas for the third graph
    canvas3 = FigureCanvasTkAgg(figure3, master=root)
    canvas3.draw()
    canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    loading_screen.withdraw()

# Create predict button
predict_button = tk.Button(root, text="Predict", command=predict_price)
predict_button.pack()

message_text = tk.Text(root, height=10, width=50, font=("Arial", 12))
message_text.pack()

root.mainloop()
