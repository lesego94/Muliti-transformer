#!/usr/bin/env python
# coding: utf-8

# In[10]:


import getFamaFrenchFactors as gff
import pandas as pd
import pypfopt
from matplotlib import pyplot as plt
from pypfopt import expected_returns
from pypfopt import risk_models
from pypfopt import plotting
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import statsmodels.tsa.stattools as st
import numpy as np
from pmdarima.arima import auto_arima
from pykalman import KalmanFilter
import getFamaFrenchFactors as gff
from sktime.forecasting.base import ForecastingHorizon
from sklearn.neighbors import KNeighborsRegressor
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import (
    ForecastingGridSearchCV,
    SlidingWindowSplitter)
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.trend import TrendForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.transformations.bootstrap import STLBootstrapTransformer
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.compose import BaggingForecaster
from sklearn.metrics import r2_score


# In[11]:


def my_plot(pre_data,test_data):
    for col in pre_data.columns:
        plt.title(col)
        plt.plot(pre_data[col], label = 'Prediction')
        plt.plot(test_data[col], label = 'Real')
        plt.legend(loc="upper right")
        plt.show()


# In[12]:


class My_Prediction_Model:
    
    def __init__(self,train_data,test_data,model):
        
        self.train=train_data
        self.test=test_data
        self.model=model
  
    #use ARMA model for predicion
    def ARMA_model(self):
        
        df_train=self.train
        df_test=self.test
        
        """
        ARMA_dict={}
        
        #use aic rule to find best parameters
        for i in df_train.columns:
            ARMA_dict[i]=sm.tsa.arma_order_select_ic(df_train[i],max_ar=3,max_ma=3,ic='aic')['aic_min_order']
            #print("AIC for"+i,ARMA_dict[i])
        """
        ARMA_pre_dict={}
        mse_dict={}
        mape_dict={}
        date_index=df_test.index
        r2score={}
        
        def acf_pacf_plot(data):
   
            sm.graphics.tsa.plot_acf(data,lags=30,title=str(data.name)+" Autocorrelation") #ARIMA,q
            sm.graphics.tsa.plot_pacf(data,lags=30,title=str(data.name)+" Partial Autocorrelation") #ARIMA,p
            return


    
        for i in df_train.columns:
            name=i
            #acf_pacf_plot(df_train[name])
            #AR_order=i[1][0]
            #MA_order=i[1][1]
            #if AR_order !=0 or MA_order !=0:
            model=auto_arima(df_train[name], start_p=0, d=0, 
                             start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=None, 
                             start_Q=1, max_P=5, max_D=5, max_Q=5, m=1, seasonal=True, 
                             stationary=False, information_criterion='aic')
            print(name)
            print(model.summary())

            forecast_n = df_test.shape[0]#forcasting months
            forecast_ARMA = model.predict(n_periods=forecast_n)
            
            ARMA_pre_dict[name]=forecast_ARMA
            #print(forecast_ARMA)
            """
            plt.title(name)
            plt.plot(date_index,df_test[name].values)
            plt.plot(date_index,forecast_ARMA, color='red')
            plt.show()
            """
            mse_dict[name]=mean_squared_error(df_test[name].values,forecast_ARMA)
            mape_dict[name]=mean_absolute_percentage_error(df_test[name].values,forecast_ARMA)
            r2score[name]=r2_score(df_test[name].values,forecast_ARMA )
            
        self.pre=pd.DataFrame(ARMA_pre_dict,index=date_index)
        self.mse=pd.DataFrame([mse_dict])
        self.mape=pd.DataFrame([mape_dict])
        self.r2=pd.DataFrame([r2score])
        
        return
    
    #use Kalman-Filted data to do ARMA predictioin
    def KF_ARMA_model(self,alpha):
        
        df_train=self.train
        df_test=self.test
        
        def Kalman1D(observations,damping=1):
            # To return the smoothed time series data
            observation_covariance = damping
            initial_value_guess = observations[0]
            transition_matrix = 1
            transition_covariance = 0.1
            initial_value_guess
            kf = KalmanFilter(
                    initial_state_mean=initial_value_guess,
                    initial_state_covariance=observation_covariance,
                    observation_covariance=observation_covariance,
                    transition_covariance=transition_covariance,
                    transition_matrices=transition_matrix
                )
            pred_state, state_cov = kf.smooth(observations)
            return pred_state
        

        KF_dict={}
        mse_dict={}
        mape_dict={}
        KF_ARMA_pre_dict={}
        r2score={}
        
        for i in df_train.columns:
            KF_dict[i]=Kalman1D(df_train[i],alpha).squeeze()
            
        df_KF=pd.DataFrame(KF_dict)
        
        date_index=df_test.index
        
         
        def acf_pacf_plot(data):
   
            sm.graphics.tsa.plot_acf(data,lags=30,title=str(data.name)+" Autocorrelation") #ARIMA,q
            sm.graphics.tsa.plot_pacf(data,lags=30,title=str(data.name)+" Partial Autocorrelation") #ARIMA,p
            return
        
        for i in df_train.columns:
            name=i
            #acf_pacf_plot(df_train[name])
            
            model=auto_arima(df_KF[name], start_p=0, d=0, 
                             start_q=0, max_p=5, max_d=5, max_q=5, start_P=0, D=None, 
                             start_Q=1, max_P=5, max_D=5, max_Q=5, m=1, seasonal=True, 
                             stationary=False, information_criterion='aic')
            print(name)
            print(model.summary())

            forecast_n = df_test.shape[0]#forcasting months
            forecast_ARMA = model.predict(n_periods=forecast_n)
            
            KF_ARMA_pre_dict[name]=forecast_ARMA
            #print(forecast_ARMA)
            """
            plt.title(name)
            plt.plot(date_index,df_test[name].values)
            plt.plot(date_index,forecast_ARMA, color='red')
            plt.show()
            """
            mse_dict[name]=mean_squared_error(df_test[name].values,forecast_ARMA)
            mape_dict[name]=mean_absolute_percentage_error(df_test[name].values,forecast_ARMA)
            r2score[name]=r2_score(df_test[name].values,forecast_ARMA )


        self.pre=pd.DataFrame(KF_ARMA_pre_dict,index=date_index)
        self.mse=pd.DataFrame([mse_dict])
        self.mape=pd.DataFrame([mape_dict])
        self.r2=pd.DataFrame([r2score])
        
        return 
    
    #use KNN for prediction
    def KNN_model(self):
        
        df_train=self.train
        df_test=self.test
        
        pred_sktimes = {}
        MAPE = {}
        MSE = {}
        r2score={}
        date_index=df_test.index
        
        for col in df_train.columns:

            y_train = df_train[col]
            y_test = df_test[col]

            y_train.index = pd.PeriodIndex(y_train.index, freq='M')
            y_test.index = pd.PeriodIndex(y_test.index, freq='M')

            fh = ForecastingHorizon(y_test.index, is_relative=False)
            regressor = KNeighborsRegressor(n_neighbors=12)
            forecaster = make_reduction(regressor, window_length=12, strategy="recursive")
            forecaster.fit(y_train)
            y_pred = forecaster.predict(fh)
            
            mean_absolute_percentage_error(y_test, y_pred)
            mean_squared_error(y_test, y_pred)

            pred_sktimes[col]=y_pred
            #print(pred_sktimes)
            MAPE[col]=mean_absolute_percentage_error(y_test, y_pred)
            MSE[col]=mean_squared_error(y_test, y_pred)
            r2score[col]=r2_score(y_test, y_pred)
        #print("..........",pred_sktimes)
        #self.pre=pd.DataFrame(pred_sktimes,index=date_index)
        df_pre=pd.DataFrame(pred_sktimes)
        df_pre.set_index(date_index,inplace=True)
        self.pre=pd.DataFrame(df_pre)
        self.mse=pd.DataFrame([MSE])
        self.mape=pd.DataFrame([MAPE])
        self.r2=pd.DataFrame([r2score])
        return
    
    #use KNN with pipeline
    def KNN_PIPE_model(self):
        
         
        df_train=self.train
        df_test=self.test
        
        pred_sktimes = {}
        MAPE = {}
        MSE = {}
        r2score={}
        date_index=df_test.index
        for col in df_train.columns:

            y_train = df_train[col]
            y_test = df_test[col]

            y_train.index = pd.PeriodIndex(y_train.index, freq='M')
            y_test.index = pd.PeriodIndex(y_test.index, freq='M')

            fh = ForecastingHorizon(y_test.index, is_relative=False)

            regressor = KNeighborsRegressor()
            forecaster = make_reduction(regressor, window_length=12, strategy="recursive")
            param_grid = {"window_length": [7, 12, 15]}

            # We fit the forecaster on an initial window which is 80% of the historical data
            # then use temporal sliding window cross-validation to find the optimal hyper-parameters
            cv = SlidingWindowSplitter(initial_window=int(len(y_train) * 0.8), window_length=20)
            gscv = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)


            gscv.fit(y_train)
            y_pred = gscv.predict(fh)
            #mean_absolute_percentage_error(y_test, y_pred)
            #mean_squared_error(y_test, y_pred)
            MAPE[col]=mean_absolute_percentage_error(y_test, y_pred)
            MSE[col]=mean_squared_error(y_test, y_pred)
            pred_sktimes[col]=y_pred
            r2score[col]=r2_score(y_test, y_pred)
            
        df_pre=pd.DataFrame(pred_sktimes)
        df_pre.set_index(date_index,inplace=True)
        self.pre=pd.DataFrame(df_pre)
        self.mse=pd.DataFrame([MSE])
        self.mape=pd.DataFrame([MAPE])
        self.r2=pd.DataFrame([r2score])
        
        return 
    
    def Trend_model(self):
        
         
        df_train=self.train
        df_test=self.test
        
        pred= {}
        MAPE = {}
        MSE = {}
        r2score={}
        date_index=df_test.index
        for col in df_train.columns:

            y_train = df_train[col]
            y_test = df_test[col]

            y_train.index = pd.PeriodIndex(y_train.index, freq='M')
            y_test.index = pd.PeriodIndex(y_test.index, freq='M')

            fh = ForecastingHorizon(y_test.index, is_relative=False)

            forecaster =TrendForecaster()
           
            forecaster.fit(y_train)
            y_pred = forecaster.predict(fh)
            #mean_absolute_percentage_error(y_test, y_pred)
            #mean_squared_error(y_test, y_pred)
            MAPE[col]=mean_absolute_percentage_error(y_test, y_pred)
            MSE[col]=mean_squared_error(y_test, y_pred)
            pred[col]=y_pred
            r2score[col]=r2_score(y_test, y_pred)
            
        df_pre=pd.DataFrame(pred)
        df_pre.set_index(date_index,inplace=True)
        self.pre=pd.DataFrame(df_pre)
        self.mse=pd.DataFrame([MSE])
        self.mape=pd.DataFrame([MAPE])
        self.r2=pd.DataFrame([r2score])

        return 
    
    def Poly_Trend_model(self):
        
         
        df_train=self.train
        df_test=self.test
        
        pred= {}
        MAPE = {}
        MSE = {}
        r2score={}
        date_index=df_test.index
        for col in df_train.columns:

            y_train = df_train[col]
            y_test = df_test[col]

            y_train.index = pd.PeriodIndex(y_train.index, freq='M')
            y_test.index = pd.PeriodIndex(y_test.index, freq='M')

            fh = ForecastingHorizon(y_test.index, is_relative=False)

            forecaster=PolynomialTrendForecaster(degree=1)
            
            param_grid = {"degree": [1,2,3]}

            # We fit the forecaster on an initial window which is 80% of the historical data
            # then use temporal sliding window cross-validation to find the optimal hyper-parameters
            cv = SlidingWindowSplitter(initial_window=int(len(y_train) * 0.8), window_length=20)
            gscv = ForecastingGridSearchCV(forecaster, strategy="refit", cv=cv, param_grid=param_grid)
            
            forecaster.fit(y_train)
            y_pred = forecaster.predict(fh)
            #mean_absolute_percentage_error(y_test, y_pred)
            #mean_squared_error(y_test, y_pred)
            MAPE[col]=mean_absolute_percentage_error(y_test, y_pred)
            MSE[col]=mean_squared_error(y_test, y_pred)
            pred[col]=y_pred
            r2score[col]=r2_score(y_test, y_pred)
            
        df_pre=pd.DataFrame(pred)
        df_pre.set_index(date_index,inplace=True)
        self.pre=pd.DataFrame(df_pre)
        self.mse=pd.DataFrame([MSE])
        self.mape=pd.DataFrame([MAPE])
        self.r2=pd.DataFrame([r2score])
        return 
    
    def Bagging_model(self):
        
         
        df_train=self.train
        df_test=self.test
        
        pred= {}
        MAPE = {}
        MSE = {}
        r2score={}
        date_index=df_test.index
        for col in df_train.columns:

            y_train = df_train[col]
            y_test = df_test[col]

            y_train.index = pd.PeriodIndex(y_train.index, freq='M')
            y_test.index = pd.PeriodIndex(y_test.index, freq='M')

            fh = ForecastingHorizon(y_test.index, is_relative=False)

            forecaster=forecaster = BaggingForecaster(STLBootstrapTransformer(sp=4), NaiveForecaster(sp=4))
        

          
            forecaster.fit(y_train)
            y_pred = forecaster.predict(fh)
            #mean_absolute_percentage_error(y_test, y_pred)
            #mean_squared_error(y_test, y_pred)
            MAPE[col]=mean_absolute_percentage_error(y_test, y_pred)
            MSE[col]=mean_squared_error(y_test, y_pred)
            pred[col]=y_pred
            r2score[col]=r2_score(y_test, y_pred)
            
        df_pre=pd.DataFrame(pred)
        df_pre.set_index(date_index,inplace=True)
        self.pre=pd.DataFrame(df_pre)
        self.mse=pd.DataFrame([MSE])
        self.mape=pd.DataFrame([MAPE])
        self.r2=pd.DataFrame([r2score])

        return 
    
    
    
    
    def model_prediction(self,myalpha=0.05,pred=True,mse=True,mape=True,graph=True):
        
      
                
        model=self.model
        
        if model=="ARMA":
            
            self.ARMA_model()
            
            if pred:
                print(model+str(" return prediction: "),self.pre)
            if mse:
                print(model+str(" mse: "),self.mse)
            if mape:
                print(model+str(" mape: "),self.mape)
            if graph:
                my_plot(self.pre,self.test)
                
        if model=="KF_ARMA":
            
            self.KF_ARMA_model(myalpha)
            
            if pred:
                print(model+str(" return prediction: "),self.pre)
            if mse:
                print(model+str(" mse: "),self.mse)
            if mape:
                print(model+str(" mape: "),self.mape)
            if graph:
                print(self.pre)
                my_plot(self.pre,self.test)
        
        if model=="KNN":
            
            self.KNN_model()
            
            if pred:
                print(model+str(" return prediction: "),self.pre)
            if mse:
                print(model+str(" mse: "),self.mse)
            if mape:
                print(model+str(" mape: "),self.mape)
            if graph:
                my_plot(self.pre,self.test)
        
        
        if model=="KNN_PIPE":
            
            self.KNN_PIPE_model()
            
            if pred:
                print(model+str(" return prediction: "),self.pre)
            if mse:
                print(model+str(" mse: "),self.mse)
            if mape:
                print(model+str(" mape: "),self.mape)
            if graph:
                
                my_plot(self.pre,self.test)
        
        
        if model=="Trend":
            
            self.Trend_model()
            
            if pred:
                print(model+str(" return prediction: "),self.pre)
            if mse:
                print(model+str(" mse: "),self.mse)
            if mape:
                print(model+str(" mape: "),self.mape)
            if graph:
                
                my_plot(self.pre,self.test)
        
        if model=="Poly_Trend":
            
            self.Poly_Trend_model()
            
            if pred:
                print(model+str(" return prediction: "),self.pre)
            if mse:
                print(model+str(" mse: "),self.mse)
            if mape:
                print(model+str(" mape: "),self.mape)
            if graph:
                
                my_plot(self.pre,self.test)
                
        if model=="Bagging":
            
            self.Bagging_model()
            
            if pred:
                print(model+str(" return prediction: "),self.pre)
            if mse:
                print(model+str(" mse: "),self.mse)
            if mape:
                print(model+str(" mape: "),self.mape)
            if graph:
                
                my_plot(self.pre,self.test)
        
        
            
            
        return


# In[ ]:





# In[ ]:





# In[ ]:




