import streamlit as st
import pandas as pd
import yfinance as yf
import datetime

st.write("""
# Цена акций компании Apple
         """)

tickerSymbol = "AAPL"
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(start = '2024-11-30', end='2025-11-22')

st.write("""
## Цена закрытия
         """)

st.line_chart(tickerDf.Close)
st.write("""
## Всего получено
         """)

st.line_chart(tickerDf.Volume)