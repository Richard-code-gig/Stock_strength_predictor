from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from mplfinance.original_flavor import candlestick_ohlc
import numpy as np
import pandas as pd
import pandas_ta as ta
import locale


locale.setlocale(locale.LC_ALL, '')

#create subplots

fig = plt.figure()
fig.patch.set_facecolor('#121416')
gs = fig.add_gridspec(6,6)
ax1 = fig.add_subplot(gs[0:4, 0:4])
ax2 = fig.add_subplot(gs[0, 4:6])
ax3 = fig.add_subplot(gs[1, 4:6])
ax4 = fig.add_subplot(gs[2, 4:6])
ax5 = fig.add_subplot(gs[3, 4:6])
ax6 = fig.add_subplot(gs[4, 4:6])
ax7 = fig.add_subplot(gs[5, 4:6])
ax8 = fig.add_subplot(gs[4, 0:4])
ax9 = fig.add_subplot(gs[5, 0:4])


stock = ['PYPL', 'AAPL', 'TWTR', 'AMZN', 'MSFT', 'FB', 'BRK-B']

def figure_design(ax):
    ax.set_facecolor('#091217')
    ax.tick_params(axis='both', labelsize=14, colors='white')
    ax.ticklabel_format(useOffset=True)
    ax.spines['bottom'].set_color('#808080')
    ax.spines['top'].set_color('#808080')
    ax.spines['left'].set_color('#808080')
    ax.spines['right'].set_color('#808080')
    
def figure_designs(ax):
    ax.set_facecolor('#091217')
    ax.tick_params(axis='both', labelsize=8, colors='white')
    ax.spines['bottom'].set_color('#808080')

#make a function to create OHLC data, Stochastic Oscillator and moving averages
def read_data_ohlc(filename, stock_code, usecols):
    df = pd.read_csv(filename, header=None, usecols=usecols, names=['time', 'Quote Price', 'change', 'volume', 'target'],
                index_col='time', parse_dates=['time'])

    index_with_nan = df.index[df.isnull().any(axis=1)]
    df.drop(index_with_nan, 0, inplace=True)
    df.index = pd.DatetimeIndex(df.index)

    latest_info = df.iloc[-1, :] 
    latest_price = str(latest_info.iloc[0])
    latest_change = str(latest_info.iloc[1])
    
    df_vol = df['volume'].resample('1Min').mean()
    data = df['Quote Price'].resample('1Min').ohlc()
    
    #create data for stochastic oscillator 
    data_sk = df['Quote Price'].resample('1Min').ohlc()
    
    # Define periods
    k_period = 14
    d_period = 3
    # Add "n_high" column with max value of previous 14 periods
    data_sk['n_high'] = data_sk['high'].rolling(k_period).max()
    # Add "n_low" column with min value of previous 14 periods
    data_sk['n_low'] = data_sk['low'].rolling(k_period).min()
    # Use the min/max values to calculate the %k
    data_sk['%K'] = (data_sk['close'] - data_sk['n_low']) * 100 / (data_sk['n_high'] - data_sk['n_low'])
    # Uses the %k to calculates a SMA over the past 3 values of %k
    data_sk['%D'] = data_sk['%K'].rolling(d_period).mean() 
    # Add some indicators
    data_skd = data_sk.ta.stoch(high='high', low='low', k=14, d=3, append=True)
    
    #Make moving avg columns from initial data
    data['time'] = data.index
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d %H:%M:%S')
    data['MA5'] = data['close'].rolling(5).mean()
    data['MA10'] = data['close'].rolling(10).mean()
    data['MA20'] = data['close'].rolling(20).mean()
    
    data['volume_diff'] = df_vol.diff()
    data[data['volume_diff']<0] = None
    
    index_with_nan = data.index[data.isnull().any(axis=1)]
    data.drop(index_with_nan, 0, inplace = True)
    data.reset_index(drop=True, inplace=True)

    return data, latest_price, latest_change, df['target'][-1], df['volume'][-1], data_skd

#Make a function to generate any additional required data
def read_data_rest(filename, stock_code, usecols):
    df = pd.read_csv(filename, header=None, usecols=usecols, names=['time', 'Quote Price', 'change', 'target'],
                index_col='time', parse_dates=['time'])
    
    index_with_nan = df.index[df.isnull().any(axis=1)]
    df.drop(index_with_nan, 0, inplace=True)
    return df

#Make a fcnction to calculate RSI
def rsi(filename, stock_code, usecols, periods = 14, ema = True):
    df = read_data_rest(filename, stock_code, usecols)
    """
    Returns a pd.Series with the relative strength index.
    """
    close_delta = df['Quote Price'].diff() #Previous close is to be used but this project deals with daily data
    close_delta.dropna(inplace=True)
    close_delta.index = pd.DatetimeIndex(close_delta.index)

    up = close_delta.clip(lower=0)
    down = -1 * close_delta.clip(upper=0)
    
    if ema == True:
	    # Use exponential moving average
        ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
    else:
        # Use simple moving average
        ma_up = up.rolling(window = periods).mean()
        ma_down = down.rolling(window = periods).mean()
        
    rsi = ma_up / ma_down
    rsi = round(100 - (100/(1 + rsi)), 2)
    df = rsi.to_frame(name='rsi')
    return df


def animate(i):
    
    data, latest_price, latest_change, target, volume, data_sk = \
    read_data_ohlc('stockdata.csv', stock[0], usecols=[0,1,2,3,4])
    
    candle_counter = range(len(data['open'])-1)
    ohlc = []
    for candle in candle_counter:
        append_me = candle_counter[candle], data['open'][candle], \
        data['high'][candle], data['low'][candle], \
        data['close'][candle]
        
        ohlc.append(append_me)
        
    ax1.clear()
    candlestick_ohlc(ax1, ohlc, width=0.4, colorup='#18b800', colordown='#ff3503')
    
    ax1.plot(data['MA5'], color='pink', linestyle='-', linewidth=1, label='5 min SMA')
    ax1.plot(data['MA10'], color='orange', linestyle='-', linewidth=1, label='10 min SMA')
    ax1.plot(data['MA20'], color='#08a0e9', linestyle='-', linewidth=1, label='20 min SMA')
    leg=ax1.legend(loc='upper left', facecolor = '#121416', fontsize=10)
    
    for text in leg.get_texts():
        plt.setp(text, color='w')
    
    figure_design(ax1)
    ax1.text(0.005, 1.05, stock[0], transform=ax1.transAxes, color='black', fontsize=18,
            fontweight='bold', horizontalalignment='left', verticalalignment='center',
            bbox=dict(facecolor='#FFBF00'))
    ax1.text(0.2, 1.05, latest_price, transform=ax1.transAxes, color='white', fontsize=18,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    if latest_change[0]=='-':
        colorcode = '#ff3503'
    else:
        colorcode = '#18b800'
        
    ax1.text(0.5, 1.05, latest_change, transform=ax1.transAxes, color=colorcode, fontsize=18,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax1.text(0.8, 1.05, target, transform=ax1.transAxes, color='#08a0e9', fontsize=18,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    time_stamp = datetime.now()
    time_stamp = time_stamp.strftime('%Y-%m-%d %H:%M:%S')
    ax1.text(1.35, 1.05, time_stamp, transform=ax1.transAxes, color='white', fontsize=12,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax1.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    
    #ax8 is stochastic oscillator, put it here
    ax8.clear()
    ax8.plot(data_sk['STOCHk_14_3_3'], color='pink', linestyle='-', linewidth=1, label='Sk')
    ax8.plot(data_sk['STOCHd_14_3_3'], color='orange', linestyle='-', linewidth=1, label='Sd')
    ax8.legend(loc='upper right', labelcolor='white', fontsize='small', frameon=False, facecolor='#121416')
    figure_designs(ax8)
    ax8.text(0.005, 0.9, '%k: ' +str(round(data_sk['STOCHk_14_3_3'][-1], 2))+', %d: '+str(round(data_sk['STOCHd_14_3_3'][-1], 2)),
            transform=ax8.transAxes, color='yellow', fontsize=8, fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax8.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    
    #ax2 to ax7 generata price plot for the other 6 stocks
    df = read_data_rest('stockdata.csv', stock_code=stock[1], usecols=[0,5,6,8])
    ax2.clear()
    ax2.plot(df['Quote Price'], color='white', linestyle='-', linewidth=1)
    ax2.set_facecolor('#091217')
    ax2.text(0.005, 0.9, stock[1], transform=ax2.transAxes, color='yellow', fontsize=8,
        fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax2.text(0.3, 0.9, str(df['Quote Price'][-1]), transform=ax2.transAxes, color='white', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    if df['change'][-1][0]=='-':
        colorcode = '#ff3503'
    else:
        colorcode = '#18b800'
        
    ax2.text(0.6, 0.9, df['change'][-1], transform=ax2.transAxes, color=colorcode, fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax2.text(0.9, 0.9, str(df['target'][-1]), transform=ax2.transAxes, color='#08a0e9', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    
    df = read_data_rest('stockdata.csv', stock[2], usecols=[0,9,10,12])
    ax3.clear()
    ax3.plot(df['Quote Price'], color='white', linestyle='-', linewidth=1)
    ax3.set_facecolor('#091217')
    ax3.text(0.005, 0.9, stock[2], transform=ax3.transAxes, color='yellow', fontsize=8,
        fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax3.text(0.3, 0.9, str(df['Quote Price'][-1]), transform=ax3.transAxes, color='white', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    if df['change'][-1][0]=='-':
        colorcode = '#ff3503'
    else:
        colorcode = '#18b800'
        
    ax3.text(0.6, 0.9, df['change'][-1], transform=ax3.transAxes, color=colorcode, fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax3.text(0.9, 0.9, str(df['target'][-1]), transform=ax3.transAxes, color='#08a0e9', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    
    df = read_data_rest('stockdata.csv', stock_code=stock[3], usecols=[0,13,14,16])
    ax4.clear()
    ax4.plot(df['Quote Price'], color='white', linestyle='-', linewidth=1)
    ax4.set_facecolor('#091217')
    ax4.text(0.005, 0.9, stock[3], transform=ax4.transAxes, color='yellow', fontsize=8,
        fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax4.text(0.3, 0.9, str(df['Quote Price'][-1]), transform=ax4.transAxes, color='white', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    if df['change'][-1][0]=='-':
        colorcode = '#ff3503'
    else:
        colorcode = '#18b800'
        
    ax4.text(0.62, 0.9, df['change'][-1], transform=ax4.transAxes, color=colorcode, fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax4.text(0.9, 0.9, str(df['target'][-1]), transform=ax4.transAxes, color='#08a0e9', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    
    df = read_data_rest('stockdata.csv', stock_code=stock[4], usecols=[0,17,18,20])
    ax5.clear()
    ax5.plot(df['Quote Price'], color='white', linestyle='-', linewidth=1)
    ax5.set_facecolor('#091217')
    ax5.text(0.005, 0.9, stock[4], transform=ax5.transAxes, color='yellow', fontsize=8,
        fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax5.text(0.3, 0.9, str(df['Quote Price'][-1]), transform=ax5.transAxes, color='white', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    if df['change'][-1][0]=='-':
        colorcode = '#ff3503'
    else:
        colorcode = '#18b800'
        
    ax5.text(0.6, 0.9, df['change'][-1], transform=ax5.transAxes, color=colorcode, fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax5.text(0.9, 0.9, str(df['target'][-1]), transform=ax5.transAxes, color='#08a0e9', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    
    df = read_data_rest('stockdata.csv', stock_code=stock[5], usecols=[0,21,22,24])
    ax6.clear()
    ax6.plot(df['Quote Price'], color='white', linestyle='-', linewidth=1)
    ax6.set_facecolor('#091217')
    ax6.text(0.005, 0.9, stock[5], transform=ax6.transAxes, color='yellow', fontsize=8,
        fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax6.text(0.3, 0.9, str(df['Quote Price'][-1]), transform=ax6.transAxes, color='white', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    if df['change'][-1][0]=='-':
        colorcode = '#ff3503'
    else:
        colorcode = '#18b800'
        
    ax6.text(0.6, 0.9, df['change'][-1], transform=ax6.transAxes, color=colorcode, fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax6.text(0.9, 0.9, str(df['target'][-1]), transform=ax6.transAxes, color='#08a0e9', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    
    df = read_data_rest('stockdata.csv', stock_code=stock[6], usecols=[0,25,26,28])
    ax7.clear()
    ax7.plot(df['Quote Price'], color='white', linestyle='-', linewidth=1)
    ax7.set_facecolor('#091217')
    ax7.text(0.005, 0.9, stock[6], transform=ax7.transAxes, color='yellow', fontsize=8,
        fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax7.text(0.3, 0.9, str(df['Quote Price'][-1]), transform=ax7.transAxes, color='white', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    if df['change'][-1][0]=='-':
        colorcode = '#ff3503'
    else:
        colorcode = '#18b800'
        
    ax7.text(0.6, 0.9, df['change'][-1], transform=ax7.transAxes, color=colorcode, fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    ax7.text(0.9, 0.9, str(df['target'][-1]), transform=ax7.transAxes, color='#08a0e9', fontsize=8,
            fontweight='bold', horizontalalignment='center', verticalalignment='center')
    
    #ax9 is for relative strength index
    df = rsi('stockdata.csv', stock[0], usecols=[0,1,2,4], periods = 14)
    ax9.clear()
    ax9.plot(df.index.values, df['rsi'], color='white', linestyle='-', linewidth=1)
    figure_designs(ax9)
    ax9.text(0.005, 0.9, f"RSI(14): {str(df['rsi'][-1])}", transform=ax9.transAxes, color='white', fontsize=8,
        fontweight='bold', horizontalalignment='left', verticalalignment='center')
    ax9.grid(True, color='grey', linestyle='-', which='major', axis='both', linewidth=0.3)
    
    xlocator = mdates.MinuteLocator(byminute=[0,15,30,45], interval = 1)
    date_form = mdates.DateFormatter("%H:%M")
    ax9.xaxis.set_minor_locator(xlocator)
    ax9.xaxis.set_major_formatter(date_form)
    
ani = FuncAnimation(fig, animate, interval=1)
plt.show()
