import os
import pandas as pd
import numpy as np
import scipy.optimize as sco
indices = ['AAPL', 'MSFT', 'NVDA', 'JNJ', 'NVS','JPM','GS','AMZN','DIS','MCD','NEE','BA','CAT','XOM','CVX','RIO','BHP']



def retrieve_var(date):
    File_path = "../Output/assets/Drop000/"
    Files = os.listdir(File_path)   
    symbol_list = []
    variance_list = []
    # Initialize an empty DataFrame to store forecasts
    Forecasts = pd.DataFrame(columns=['Asset', 'predicted var'])

    # Loop through all files

    for file in Files:
        # Skip the ".DS_Store" file
        if file == ".DS_Store":
            continue
        
        path = os.path.join(File_path, file)
        asset_name = file.split('.')[0].split('_')[3]  # Assumes file name is the asset name
        symbol_list.append(asset_name)
        asset = pd.read_csv(path)
        
        # Get 'VaR_T_ANN_ARCH' for the given date
        var_value = asset[asset['Date_Forecast'] == date]['Forecast_T_ANN_ARCH'].values[0]
        variance_list.append(var_value)
        
    variance_list = np.reshape(variance_list, (1, -1))
    # Create a DataFrame with the asset names as columns and predicted variances as the row
    df = pd.DataFrame(variance_list, columns=symbol_list, index=['predicted var'])
    return df



def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    
    # Recall portfolio_annualised_performance(weights, mean_returns, cov_matrix) returns portfolio standard deviation and portfolio return
    p_var = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_ret = np.sum(mean_returns*weights)
    return -(p_ret - risk_free_rate/52) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0,0.25)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def MVO_result(df,mean_returns, cov_matrix, risk_free_rate):    

    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    print ("-"*80)
    print ("Maximum Sharpe Ratio Portfolio Allocation\n")
    print (max_sharpe)
    
    weights = max_sharpe['x']
    rp = np.sum(mean_returns*weights)
    sdp = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    
    max_sharpe_allocation = pd.DataFrame(max_sharpe.x,index=df.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    print ("-"*80)
    print ("Weekly Return:", round(rp,5))
    print ("Weekly Volatility:", round(sdp,5))
    print ("Max Weekly Sharpe Ratio:", (rp - (risk_free_rate/52))/sdp)
    print ("\n")
    print (max_sharpe_allocation)
    return max_sharpe.x