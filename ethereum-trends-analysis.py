#!/usr/bin/env python
# coding: utf-8

# # Introduction

# **Ethereum** is a blockchain platform with its own cryptocurrency, called **Ether** (ETH) or Ethereum, and its own programming language, called Solidity.
# 
# As a blockchain network, Ethereum is a **decentralized** public ledger for verifying and recording transactions. The network's users can create, publish, monetize, and use **applications** on the platform, and use its Ether cryptocurrency as payment. Insiders call the decentralized applications on the network "dapps."
# 
# As a cryptocurrency, Ethereum is second in market value only to Bitcoin, as of May 2021.
# 
# Before we start our analysis I would like to thank @Arpit Verma for sharing this dataset

# In[ ]:


get_ipython().system('pip install mplfinance')


# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates


# In[4]:


df = pd.read_csv("ETH-USD.csv",parse_dates=True)


# In[5]:


df = df.drop(columns=['Adj Close'])


# In[6]:


df.head()


# In[29]:


df.tail()
#just wanted to see end dates or date-range [08-2015 -> 08-2021]


# In[30]:


df.describe()


# In[31]:


df['100ma'] = df['Close'].rolling(window = 100, min_periods = 0).mean()


# In[32]:


df


# # Visualizations

# In[33]:


fig, ax = plt.subplots(figsize=(16,6))
ax.plot(df.Date, df.Close)
ax.plot(df.Date, df['100ma'])
ax.xaxis.set_major_locator(plt.MaxNLocator(15)) # reduce number of x-labels
plt.title('Ethereum Prices')
plt.grid()
plt.show()


# In[34]:


fig, ax = plt.subplots(figsize=(16,6))
ax.plot(df.Date, df.Volume)
ax.xaxis.set_major_locator(plt.MaxNLocator(15)) # reduce number of x-labels
plt.title('Ethereum Volumes')
plt.grid()
plt.show()


# # Market Cap

# In[35]:


df['Total Traded'] = df['Open']*df['Volume']


# In[36]:


fig, ax = plt.subplots(figsize=(16,6))
ax.plot(df.Date, df['Total Traded'])
ax.xaxis.set_major_locator(plt.MaxNLocator(15)) # reduce number of x-labels
plt.title('Total Traded')
plt.grid()
plt.show()


# We notice a **huge spike** in Ethereum market cap somewhere in 2021 let's investigate more :

# In[37]:


df.iloc[df['Total Traded'].argmax()]


# After a quick web search we found out that around 13th March, Ethereum spiked due to the rise of NFTs and DeFi applications to quote from the article : 
# > Ethereum’s growth is attributed to an increasing number of developers building **decentralized finance** (DeFi) applications on the Ethereum blockchain platform. The rise of **non-fungible tokens** (NFT) also increases demand for Ethereum.
# > Lastly, continued institutional interest in treating crypto like any other security also helps ETH. Coinbase is the new NYSE. ETH is No. 2 on Coinbase after Bitcoin.
# > Seeing how the DeFi and NFT movements are directly connected to Ethereum, their increased activity has been the main driver for ETH prices in 2021, says Andrew Moss, GSR Capital’s managing director. “Users need ETH to interact with these technologies, so the more people who are involved with these communities and protocols, the higher the value of ETH goes,” he says.
# 
# * Source : https://www.forbes.com/sites/kenrapoza/2021/05/16/ethereum-faces-weekend-rout-but-some-see-eth-doubling-from-here/?sh=6845bee93817

# We are going to plot a candlestick plot for Ethereum prices from April 2021

# In[38]:


ohlc = df[(df['Date'] > '2021-04-01') & (df['Date'] <= '2021-07-26')]
ohlc = ohlc.loc[:, ['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
ohlc = ohlc.astype(float)
fig, ax = plt.subplots(figsize = (16,6))
candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)
ax.set_xlabel('Date')
ax.set_ylabel('Price')
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.show()


# # Daily Percentage Change

# The simple daily percentage change in closing price (without dividends and other factors) is the percentage change in the value of a stock over a single day of trading. It is defined by the following formula:
# 
# $$r_{t}=\frac{p_{t}}{p_{t-1}}-1$$
# 
# It's really useful as it indicates how **volatile** the currency is

# In[39]:


df["returns"] = (df["Close"]/df["Close"].shift(1)) - 1


# In[40]:


fig, ax = plt.subplots(figsize=(16,6))
ax.plot(df.Date, df['returns'])
ax.xaxis.set_major_locator(plt.MaxNLocator(15)) # reduce number of x-labels
plt.title('Returns')
plt.grid()
plt.show()


# In[41]:


df["returns"].hist(bins=100)


# In[42]:


df["returns"].describe()


# In[43]:


df["returns"].plot(kind = "box", figsize = (6,6))


# # Cumulative Return

# A cumulative return on an investment is the aggregate amount that the investment has gained or lost over time, independent of the amount of time involved. The cumulative return is expressed as a percentage, and it is the raw mathematical return of the following calculation: 
# 
# $$i_{t}=\left(1+r_{t}\right) i_{t-1}$$

# In[44]:


df["Cumulative Return"] = (1 + df["returns"]).cumprod()


# In[45]:


df


# In[46]:


fig, ax = plt.subplots(figsize=(16,6))
ax.plot(df.Date, df['Cumulative Return'])
ax.xaxis.set_major_locator(plt.MaxNLocator(15)) # reduce number of x-labels
plt.title('Cumulative Return')
plt.grid()
plt.show()


# In[47]:


df.iloc[df['Cumulative Return'].argmax()]


# # **Conclusion** : 
# 
# We see that **May 2021** was the best time to sell if you want the most profit after that the value of Ethereum kinda went down, also **2017** was also a good time for Ethereum, if you have been investing since the start you would have gained **730%** in returns and if you sold at the peak you would have gained **1300%**, We also notice that Ethereum is highly **volatile** so one should be very cautious while investing.
# 
# Cryptocurrency in general are quite volatile this is the case for many reasons and as I'm not a financial expert I'm gonna quote an article i searched for :
# >Many of the reasons for price volatility in mainstream markets hold true for cryptocurrencies as well. News developments and speculation are responsible for fueling price swings in crypto and mainstream markets alike. But their effect is exaggerated in crypto markets as they have less liquidity than traditional financial markets — a result of crypto markets lacking a robust ecosystem of institutional investors and large trading firms. Heightened volatility and a lack of liquidity can create a dangerous combination because both feed off of each other. Other than bitcoin, most other cryptocurrencies also lack established and widely adopted derivatives markets. Under the sway of day traders and speculators, crypto prices sometimes exhibit healthy volatility of the type we see in mainstream markets.
# 
# * Source : https://www.gemini.com/cryptopedia/volatility-index-crypto-market-price
# 
# But I still think that Ethereum still has a bright future especially after the recent developments of **Ethereum 2.0**

# # Why can't we forecast future prices ?

# There is far more going on in the stock market data than can be captured simply by looking at a univariate series of historical values. The stock prices are not the result of a couple of underlying causal factors, but a rather a multitude of contributions as well as a good dose of human irrationality. Indeed it has been posited that stock data is almost random.
# 
# There is a great notebook that explains why predicting stock prices is very hard : https://www.kaggle.com/carlmcbrideellis/lstm-time-series-stock-price-prediction-fail
# 
# as we can see in the acf and pacf plots below there isn't much correlations between current returns and past data
# 

# In[48]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pandas.plotting import register_matplotlib_converters


# In[49]:


acf_plot = plot_acf(df.returns.dropna(), lags=35)


# In[50]:


pacf_plot = plot_pacf(df.returns.dropna())

