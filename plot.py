from datasplit_eth import *
from prediction_function_eth import *
from matplotlib import pyplot as plt
import pandas as pd


date_time = raw_data['Date']
date_time = pd.to_datetime(date_time)
# df = df.set_index(date_time)


pred_dataframe = pd.DataFrame()
pred_dataframe['Train Predictions'] = train_predict
# pred_dataframe['Test Predictions'] = test_predict
pred_dataframe['Date'] = date_time

plt.gcf().autofmt_xdate()

plt.plot(pred_dataframe)

plt.show()

print(hello)

plt.plot(hello, 'r-x')
