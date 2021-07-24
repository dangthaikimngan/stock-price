import streamlit as st
import investpy
import datetime
import cufflinks as cf
import streamlit as st
from datetime import date
from plotly import graph_objs as go
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

from vnquant.DataLoader import DataLoader
from sklearn.metrics import mean_squared_error
import math
import seaborn as sns
import statsmodels.api as sm
import itertools
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tools.eval_measures import rmse
import warnings

from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier

from PIL import Image

page = st.sidebar.selectbox('Choose a page', ['Home', 'Portfolio', 'Forecast', 'Result'])

#                                                   Home
if page == 'Home':
    st.write("""
    # Stock Market Web Application
    **Visually** Show data on a stock!
    """)
    image = Image.open("stock-prices-up.png")
    st.image(image, use_column_width=True)


#                                                   Portfolio
if page =='Portfolio':
    st.title('Portfolio')
    stocks = ('BID','CTG', 'HDB', 'MBB','STB','TCB','TPB','VCB','VPB')
    selected_stock = st.selectbox('Select dataset for prediction', stocks)
    data_load_state = st.text('Load data...')
    if selected_stock == 'BID':
        df = investpy.get_stock_historical_data(stock='BID',country='vietnam',from_date='04/06/2018',to_date='04/06/2021')
    elif selected_stock == 'CTG':
        df = investpy.get_stock_historical_data(stock='CTG',country='vietnam',from_date='04/06/2018',to_date='04/06/2021')
    elif selected_stock == 'MBB':
        df = investpy.get_stock_historical_data(stock='MBB',country='vietnam',from_date='04/06/2018',to_date='04/06/2021')
    elif selected_stock == 'STB':
        df = investpy.get_stock_historical_data(stock='STB',country='vietnam',from_date='04/06/2018',to_date='04/06/2021')
    elif selected_stock == 'TCB':
        df = investpy.get_stock_historical_data(stock='TCB',country='vietnam',from_date='04/06/2018',to_date='04/06/2021')
    elif selected_stock == 'TPB':
        df = investpy.get_stock_historical_data(stock='TPB',country='vietnam',from_date='04/06/2018',to_date='04/06/2021')
    elif selected_stock == 'VCB':
        df = investpy.get_stock_historical_data(stock='VCB',country='vietnam',from_date='04/06/2018',to_date='04/06/2021')
    elif selected_stock == 'VPB':
        df = investpy.get_stock_historical_data(stock='VPB',country='vietnam',from_date='04/06/2018',to_date='04/06/2021')

    data_load_state.text('Loading data...done!')

    st.header('Raw data')
    st.write(df.tail())

    ## plot data
    def plot_raw_data():
        fig_1 = go.Figure()
        #fig_1.add_trace(go.Scatter(x=df.index, y=df['Open'], name = 'stock_open'))
        fig_1.add_trace(go.Scatter(x=df.index, y=df['Close'], name = 'Stock Close'))
        fig_1.layout.update(title_text = 'Time Series Data', xaxis_rangeslider_visible = True)
        st.plotly_chart(fig_1)
    plot_raw_data()

### show portfolio
    st.header('Show the portfolio')
    df_port = pd.read_excel('VN30.xlsx')
    df_port = df_port.drop(columns=['BVH', 'FPT', 'GAS', 'HPG', 'KDH', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'PNJ', 'POW', 'REE', 'SBT', 'SSI', 'TCH', 'VHM', 'VIC', 'VJC', 'VNM', 'VRE'])
    st.write(df_port.tail())
    mu = expected_returns.mean_historical_return(df_port)
    S = risk_models.sample_cov(df_port)
    ef = EfficientFrontier(mu, S)
    weight = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    st.write(cleaned_weights)
    ef.portfolio_performance(verbose=True)

    st.write('''**Porfolio include:** STB 66.59% and VCB 33.41%''')

#                                                   Forecast ARIMA
if page == 'Forecast':
    st.title('Forecast')
    stocks_1 = ('STB','VCB')
    selected_stock_1 = st.selectbox('Select dataset for prediction', stocks_1)

    if selected_stock_1 == 'STB':
        loader = DataLoader(symbols="STB", start='2016-06-01', end='2021-06-01', minimal=True, data_source='vnd')
        data = loader.download()
        data = data.droplevel(1, axis=1)
        data.columns.name = None

        data_1 = data[['high', 'low', 'open', 'close', 'avg', 'volume']]
        st.write(data_1.tail())
        train_data_1, test_data_1 = data_1[:int(len(data_1)*0.8)], data_1[int(len(data_1)*0.8):]

        def plot_stb_price():
            fig_2 = go.Figure()
            fig_2.add_trace(go.Scatter(x=data_1[:int(len(data_1)*0.8)].index, y=data_1['close'], name = 'Training Data'))
            fig_2.add_trace(go.Scatter(x=data_1[int(len(data_1)*0.8):].index, y=test_data_1['close'], name = 'Testing Data'))
            fig_2.layout.update(title_text = 'STB Prices', xaxis_rangeslider_visible = True)
            st.plotly_chart(fig_2)
        plot_stb_price()

        st.header('Forecast Processing')
        data_load_state = st.text('Load data...')

        train_model_1 = train_data_1['close'].values
        test_model_1 = test_data_1['close'].values

        history = [x for x in train_model_1]
        predictions = list()
        for t in range(len(test_model_1)):
            model = ARIMA(history, order=(5,1,0))    
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test_model_1[t]
            history.append(obs)
        error = mean_squared_error(test_model_1, predictions)
        
        data_load_state.text('Loading data...done!')
        sns.set()
        fig = plt.figure(figsize=(8,4))
        plt.plot(data_1['close'], 'green', color='blue', label='Training Data')
        plt.plot(test_data_1.index, predictions, color='green', marker='o', linestyle='dashed', label='Predicted Price')
        plt.plot(test_data_1.index, test_data_1['close'], color='red', label='Actual Price')
        plt.title('STB Prices Prediction')
        plt.xlabel('Years')
        plt.ylabel('Prices')
        st.pyplot(fig, scale = True)

        st.subheader('Predict Stock Price')
        result_data = pd.DataFrame()
        result_data['Name'] = ['Ngân hàng TMCP Sài Gòn Thương Tín']
        result_data['Ticker'] = ['STB']
        result_data['Close Price'] = math.floor(yhat)*1000
        result_data.set_index('Name')
        st.write(result_data)

# ################################### VCB
    # load the data
    if selected_stock_1 == 'VCB':
        loader = DataLoader(symbols="VCB", start='2016-06-01', end='2021-06-01', minimal=True, data_source='vnd')
        data_2 = loader.download()
        data_2 = data_2.droplevel(1, axis=1)
        data_2.columns.name = None

        data_2 = data_2[['high', 'low', 'open', 'close', 'avg', 'volume']]
        st.write(data_2.tail())
        train_data_2, test_data_2 = data_2[:int(len(data_2)*0.8)], data_2[int(len(data_2)*0.8):]

        def plot_vcb_price():
            fig_4 = go.Figure()
            fig_4.add_trace(go.Scatter(x=data_2[:int(len(data_2)*0.8)].index, y=data_2['close'], name = 'Training Data'))
            fig_4.add_trace(go.Scatter(x=data_2[int(len(data_2)*0.8):].index, y=test_data_2['close'], name = 'Testing Data'))
            fig_4.layout.update(title_text = 'VCB Prices', xaxis_rangeslider_visible = True)
            st.plotly_chart(fig_4)
        plot_vcb_price()

        st.header('Forecast Processing')
        data_load_state = st.text('Load data...')
        train_model_2 = train_data_2['close'].values
        test_model_2 = test_data_2['close'].values

        history_2 = [j for j in train_model_2]
        predictions_2 = list()
        for t in range(len(test_model_2)):
            model_2 = ARIMA(history_2, order=(5,1,0))    
            model_fit_2 = model_2.fit(disp=0)
            output_2 = model_fit_2.forecast()
            yhat_2 = output_2[0]
            predictions_2.append(yhat_2)
            obs_2 = test_model_2[t]
            history_2.append(obs_2)
        error = mean_squared_error(test_model_2, predictions_2)

        data_load_state.text('Loading data...done!')
        sns.set()
        fig = plt.figure(figsize=(8,4))
        plt.plot(data_2['close'], 'green', color='blue', label='Training Data')
        plt.plot(test_data_2.index, predictions_2, color='green', marker='o', linestyle='dashed', label='Predicted Price')
        plt.plot(test_data_2.index, test_data_2['close'], color='red', label='Actual Price')
        plt.title('VCB Prices Prediction')
        plt.xlabel('Years')
        plt.ylabel('Prices')
        st.pyplot(fig, scale = True)

        st.subheader('Predict Stock Price')
        result_data = pd.DataFrame()
        result_data['Name'] = ['Ngân hàng TMCP Ngoại Thương Việt Nam']
        result_data['Ticker'] = ['VCB']
        result_data['Close Price'] = math.floor(yhat_2)*1000
        result_data.set_index('Name')
        st.write(result_data)
#                                                   Result
if page == 'Result':
    import re
    st.title("""An Investment Portfolio""")
    portfolio_size = st.number_input('Vui lòng nhập số tiền bạn muốn đầu tư (Chỉ được nhập số!): ')
    st.button('Submit')

#STB
    data_load_state = st.text('Load data...')
    loader = DataLoader(symbols="STB", start='2016-06-01', end='2021-06-01', minimal=True, data_source='vnd')
    data = loader.download()
    data = data.droplevel(1, axis=1)
    data.columns.name = None
    data_1 = data[['high', 'low', 'open', 'close', 'avg', 'volume']]
    train_data_1, test_data_1 = data_1[:int(len(data_1)*0.8)], data_1[int(len(data_1)*0.8):]
    train_model_1 = train_data_1['close'].values
    test_model_1 = test_data_1['close'].values

    history = [x for x in train_model_1]
    predictions = list()
    for t in range(len(test_model_1)):
        model = ARIMA(history, order=(5,1,0))    
        model_fit = model.fit(disp=0)
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_model_1[t]
        history.append(obs)
    error = mean_squared_error(test_model_1, predictions)

#VCB
    loader = DataLoader(symbols="VCB", start='2016-06-01', end='2021-06-01', minimal=True, data_source='vnd')
    data_2 = loader.download()
    data_2 = data_2.droplevel(1, axis=1)
    data_2.columns.name = None
    data_2 = data_2[['high', 'low', 'open', 'close', 'avg', 'volume']]
    train_data_2, test_data_2 = data_2[:int(len(data_2)*0.8)], data_2[int(len(data_2)*0.8):]

    train_model_2 = train_data_2['close'].values
    test_model_2 = test_data_2['close'].values

    history_2 = [j for j in train_model_2]
    predictions_2 = list()
    for t in range(len(test_model_2)):
        model_2 = ARIMA(history_2, order=(5,1,0))    
        model_fit_2 = model_2.fit(disp=0)
        output_2 = model_fit_2.forecast()
        yhat_2 = output_2[0]
        predictions_2.append(yhat_2)
        obs_2 = test_model_2[t]
        history_2.append(obs_2)
    error = mean_squared_error(test_model_2, predictions_2)

    stb_weight = (portfolio_size*66.59/100)/yhat
    vcb_weight = (portfolio_size*33.41/100)/yhat_2
    data_load_state.text('Loading data...done!')

    st.markdown('Với số tiền bạn có, bạn có thể đầu tư được tổng số cổ phiếu mỗi loại là:')
    result = pd.DataFrame()
    result['Name'] = ['Amount']
    result['STB'] = math.floor(stb_weight)
    result['VCB'] = math.floor(vcb_weight)
    st.write(result)






