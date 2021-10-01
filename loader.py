from yahoo_fin.stock_info import get_quote_table 
from datetime import datetime
import pytz
import pandas as pd

#Create function to read realtime date using yahoo_finance
def read_data(stock):
    prices = get_quote_table(stock)
    df = pd.DataFrame(prices, index=[0])
    df = df[['1y Target Est', 'Quote Price', 'Previous Close', 'Volume']]
    df['change'] = df['Quote Price'] - df['Previous Close']
    #The change column doesn't appear like e.g +0.23 (+1.45%)
    #So getting it to be like that
    
    df['change2'] = (df['change'] / df['Previous Close'])*100
    df['change'] = df['change'].round(2)
    df['change2'] = df['change2'].round(2)
    
    for i, v in zip(df['change'], df['change2']):
        df['sign_change'] = '+' + str(i) if not str(i).startswith('-') else str(v)
        df['sign_change2'] = '+' + str(v) if not str(v).startswith('-') else str(v)
    df['Quote Price'] = df['Quote Price'].round(4)
    df['newchange'] = df['sign_change'] + ' '+ "(" + df['sign_change2'].astype(str) + "%)"
    
    df = df.drop(columns = ['Previous Close', 'change', 'change2', 'sign_change', 'sign_change2'])
    df.columns = ['target', 'Quote Price', 'volume', 'change']
    df = df[['Quote Price', 'change', 'volume', 'target']]
    df_list = df.values.tolist()
    
    return df_list[0][0], df_list[0][1], df_list[0][2], df_list[0][3]

#Create a while loop that will eternally collect and save data to a csv file unless stopped
stock = ['PYPL', 'AAPL', 'TWTR', 'AMZN', 'MSFT', 'FB', 'BRK-B']
while True:
    info = []
    col = []
    eastern = pytz.timezone('US/Eastern')
    time_stamp = datetime.now(eastern)
    EST_datetime_timestamp = time_stamp.strftime('%Y-%m-%d %H:%M:%S')
    
    for i in stock:
        price, change, volume, target = read_data(i)
        info.append(price)
        info.append(change)
        info.append(volume)
        info.append(target)
    
    col = [EST_datetime_timestamp]
    col.extend(info)
    df = pd.DataFrame(col)
    output = 'stockdata.csv'
    df = df.T
    df.to_csv(output, mode='a', sep=',', header=False, index=False)