import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import math
from sklearn.metrics import mean_squared_error
from datasplit_bitcoin import *
def error_calculation(model):
    # EVALUATION ON TEST DATA
    model.evaluate(x_test, y_test)
    
    trainpredict = model.predict(x_train)
    testpredict = model.predict(x_test)
    
    #TRANSFORMING DATA BACK TO ORIGINAL FORM
    train_predict = scaler.inverse_transform(trainpredict)
    test_predict = scaler.inverse_transform(testpredict)
    actual_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
    actual_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
    
    #Calculate RMSE performance metrics
    
    train_rmse = math.sqrt(mean_squared_error(actual_ytrain,train_predict))
    
    #Test Data RMSE
    test_rmse = math.sqrt(mean_squared_error(actual_ytest,test_predict))
    def MAPE(Y_actual,Y_Predicted):
        mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
        return mape
    LR_MAPE= MAPE(actual_ytrain,train_predict)
    return


def prediction_function(model,pred_days): 
    #Converting data back to original form 
    new_y = scaler.inverse_transform(y_test.reshape(-1,1))
    
    # Predicting for the next n 'days'
    
    test_index = len(test_data) - steps
    new_list = []
    x_input=test_data[test_index:].reshape(1,-1)
    
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    
    # PREDICTION FUNCTION FOR N DAYS
    def future_prediction(x_input, temp_input, steps, pred_days):
      lst_output=[]
      i=0
      while(i<pred_days):
          
          if(len(temp_input)>steps):
              x_input=np.array(temp_input[1:])
              x_input=x_input.reshape(1,-1)
              x_input = x_input.reshape((1, steps, 1))
              next_day = model.predict(x_input)
              temp_input.extend(next_day[0].tolist())
              temp_input=temp_input[1:]
              lst_output.extend(next_day.tolist())
              i=i+1 
          else:
              x_input = x_input.reshape((1,steps,1))
              next_day= model.predict(x_input)
              temp_input.extend(next_day[0].tolist())
              lst_output.extend(next_day.tolist())
              i=i+1
      return lst_output
    
    lst_output = future_prediction(x_input, temp_input, steps, pred_days)
    
    return lst_output

 
def model_selection(a,b):
    if a == 'btc_lstm':
        model = tf.keras.models.load_model('lstmmodel_btc')
    elif a == 'btc_gru':
        model = tf.keras.models.load_model('grumodel_btc')
    else:
        model = NULL
        
    error_calculation(model)
    lst_output = prediction_function(model,b)  
    plot_function(model, lst_output, b)
    return

def plot_function(model, lst_output, pred_days):
    trainpredict = model.predict(x_train)
    testpredict = model.predict(x_test)
    #TRANSFORMING DATA BACK TO ORIGINAL FORM
    train_predict = scaler.inverse_transform(trainpredict)
    test_predict = scaler.inverse_transform(testpredict)
    actual_ytrain = scaler.inverse_transform(y_train.reshape(-1,1))
    actual_ytest = scaler.inverse_transform(y_test.reshape(-1,1))
    pred_output = scaler.inverse_transform(lst_output)
    date_time = raw_data['Date']
    date_time = pd.to_datetime(date_time).dt.date
    date = raw_data['Date'].tail(1)
    new_date = []
    for  k in range(pred_days):
        new_date.extend((pd.to_datetime(date).dt.date)+pd.DateOffset(days = k))
    plt.title('Cryptocurrency Data Plot')
    plt.plot(date_time, raw_data['Close'], 'k', label = 'Actual Prices')
    plt.gcf().autofmt_xdate()
    excess = raw_data.shape[0] - train_predict.shape[0] - test_predict.shape[0]    
    plt.plot(date_time[excess + train_predict.shape[0] : excess + train_predict.shape[0] + test_predict.shape[0]], test_predict[:], 'b', label = 'Performance on Test Data')
    plt.xlabel('Date')
    plt.ylabel('Price($)')
    plt.show()
    plt.plot(new_date, pred_output, 'y', label = 'Predicted Output')
    plt.legend()
    plt.show()  
    
    return