import plotly.graph_objects as go
import pandas as pd
import numpy as np
import streamlit as st
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

def plotstocks(df):
    """Plot the stocks in the dataframe df"""
    figure = go.Figure()
    alpha=0.3
    lw=1
    for stock in df.columns.values:
        if stock == 'portfolio':
            alpha=1
            lw = 3
        else:
            alpha=0.3
            lw=1
        figure.add_trace(go.Scatter(
            x=df.index.values,
            y=df[stock],
            name=stock,
            mode='lines',
            opacity=alpha,
            line={'width': lw}
        ))
    figure.update_layout(height=600,width=800,
                         xaxis_title='Date',
                         yaxis_title='Relative growth %',
                         title='Relative Growth of optimized portfolio')
    figure.update_layout(xaxis=dict(rangeslider=dict(visible=True)))
    return figure
