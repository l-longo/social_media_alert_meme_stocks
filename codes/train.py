import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
import scipy.stats

path_data = 'your_path'
path_financial = 'your_path'
path_AR = 'your_path'

target = 'returns' #it can also be volumes



ar1 = []
#HYPERPARAMETERS
I = 360
network_days = 20
red = 10
L1 = 100
h_star = 10

#fix dates:
numdays = 546
base = datetime.date.fromisoformat('2020-12-01')


 
Ticker = ['gme', 'amc', 'nvda', 'bb', 'koss', 'nok', 'tsla', 'aapl', 'msft']

Ticker = ['tsla', 'gme', 'amc', 'nok']


AR1 = {}
ALERTS = {}
i = 0



for ticker in Ticker:
    
    ar1 = []
    
    #SUBMISSIONS AND COMMENTS RELATED TO THE TICKER:
    df = pd.read_csv(path_data + ticker + '_reddit.csv')
    df = df.sort_values(by='utc')
    df.index = pd.to_datetime(df['utc'], unit = 's').dt.date
    df.index = pd.to_datetime(df.index, format= '%d/%m/%Y')
    
    #SUBMISSIONS AND COMMENTS RELATED TO THE TICKER:
    df_subm = pd.read_csv(path_data + ticker + '_subm_reddit.csv')
    df_subm = df_subm.sort_values(by='utc')
    df_subm.index = pd.to_datetime(df_subm['utc'], unit = 's').dt.date
    df_subm.index = pd.to_datetime(df_subm.index, format= '%d/%m/%Y')
    
    #ALL SUBMISSIONS WALLSTREETBETS:
    df_all = pd.read_csv(path_data + 'total_subm.csv')
    df_all = df_all.sort_values(by='utc')
    df_all.index = pd.to_datetime(df_all['utc'], unit = 's').dt.date
    df_all.index = pd.to_datetime(df_all.index, format= '%d/%m/%Y')
    
    
    df_subm['count_subm'] = np.ones(df_subm.shape[0]) #count number of submissions per day
    df_all['count_subm'] = np.ones(df_all.shape[0]) #count number of submissions per day
    
    
    delta_test = datetime.timedelta(days=2)
    delta_train = datetime.timedelta(days=I)
    date_list = [base + datetime.timedelta(days=x) for x in range(numdays)]
    Alert0 = {}
    
    #Store results in a new dataframe:
    df_results = pd.DataFrame(np.zeros(numdays), index = date_list, columns = ['alert0'])
    df_results['alert1'] = (np.zeros(numdays))
    
    
    
              
    ####################################
    ###############STEP 1###############
    ####################################     
    
    t = 0
    for dates in date_list:
        t=t+1
        
        df_training_subm = df_subm[df_subm.index > pd.to_datetime(dates)]
        df_training_subm = df_training_subm[df_training_subm.index < pd.to_datetime(dates) + delta_train]
        
        df_test_subm = df_subm[df_subm.index > pd.to_datetime(dates)]
        df_test_subm = df_test_subm[df_test_subm.index < pd.to_datetime(dates) + delta_test]
        
        
        
        
        
        df_training_all = df_all[df_all.index > pd.to_datetime(dates)]
        df_training_all = df_training_all[df_training_all.index < pd.to_datetime(dates) + delta_train]
        
        df_test_all = df_all[df_all.index > pd.to_datetime(dates)]
        df_test_all = df_test_all[df_test_all.index < pd.to_datetime(dates) + delta_test]
        
        
        
        df_results.alert0[dates+datetime.timedelta(days=1)] = 0 
        
        if df_training_subm.empty == False and df_test_subm.empty == False:
            
            #ABSOLUTE MEASURES:
            threshold_score = df_training_subm.score.mean() + abs(df_training_subm.score-df_training_subm.score.mean()).mean()
            threshold_subm = df_training_subm.count_subm.mean() + abs(df_training_subm.count_subm-df_training_subm.count_subm.mean()).mean()
            threshold_ncomm = df_training_subm.num_comments.mean() + abs(df_training_subm.num_comments-df_training_subm.num_comments.mean()).mean()
            
            test_score = df_test_subm.score.mean()
            test_subm = df_test_subm.count_subm.mean()
            test_ncomm = df_test_subm.num_comments.mean()
            
            #RELATIVE MEASURES:
            threshold_score_all = df_training_subm.score.mean() 
            threshold_subm_all = df_training_subm.count_subm.mean() 
            threshold_ncomm_all = df_training_subm.num_comments.mean() 
            
            den_score = df_training_all.score.mean()  
            den_subm = df_training_all.count_subm.mean()  
            den_ncomm = df_training_all.num_comments.mean() 
            
            
            
            thr_score = threshold_score_all/den_score 
            thr_subm = threshold_score_all/den_subm 
            thr_ncomm = threshold_score_all/den_ncomm 
            
            
            test_score_all =  df_test_all.score.mean()/df_test_all.score.mean() 
            test_subm_all =  df_test_all.count_subm.mean()/df_test_all.count_subm.mean() 
            test_ncomm_all =  df_test_all.num_comments.mean()/df_test_all.num_comments.mean() 
            
            
            
            if (test_score > threshold_score or test_subm > threshold_subm or test_ncomm > threshold_ncomm) and (test_score_all > thr_score or test_subm_all > thr_subm  or test_ncomm_all > thr_ncomm):
                df_results.alert0[dates+datetime.timedelta(days=1)] = 1
            
                
                
                
    ####################################
    ###############STEP 2###############
    ####################################            
    
    delta_network = datetime.timedelta(days=network_days)     
            
    for t in range(df_results.shape[0]):
        if df_results['alert0'].iloc[t] == 1:
            dates = df_results.index[t]
            
            df_network = df_subm[df_subm.index < pd.to_datetime(dates)]
            df_network = df_network[df_network.index > pd.to_datetime(dates) - delta_network]
            
            author_comments = df_network.groupby('author')['num_comments'].sum().reset_index()
            author_comments = author_comments[author_comments['author'] != '[deleted]']
            author_comments['rank'] = author_comments['num_comments'].rank(ascending=False, method='min')
            author_comments = author_comments.sort_values('rank')
            
            
            df_test = df_subm[df_subm.index == pd.to_datetime(dates)]
            df_test = df_test[df_test.index < pd.to_datetime(dates) + delta_test]
            
            author_comments_test = df_test.groupby('author')['num_comments'].sum().reset_index()
            author_comments_test = author_comments_test[author_comments_test['author'] != '[deleted]']
            author_comments_test['rank'] = author_comments_test['num_comments'].rank(ascending=False, method='min')
            author_comments_test = author_comments_test.sort_values('rank')
            
            first_authors = author_comments['author'].head(red).tolist()
            first_authors_test = author_comments_test['author'].head(red).tolist()
            
            # Checking for overlap
            overlap = any(author in first_authors for author in first_authors_test)
            overlapping_authors = [author for author in first_authors_test if author in first_authors]
            
            if overlapping_authors != []:
                df_results['alert1'].iloc[t] = 1
                print(dates)
          
    
            
    
    
    ####################################
    ########IMPORT FINANCIAL DATA#######
    #################################### 
    
    financial_df = pd.read_csv(path_financial + ticker + '.csv', index_col = 0)    
    financial_df.index = pd.to_datetime(financial_df.index)
    
    crsp_df = pd.read_csv(path_financial + 'CRSP.csv', index_col = 0)    
    crsp_df.index = pd.to_datetime(crsp_df.index)
    
    crsp_df = crsp_df.sort_index()
    
    financial_df['date'] = financial_df.index
    crsp_df['date'] = crsp_df.index
    
    financial = pd.merge(financial_df, crsp_df, on='date', how='left')
    financial.index = financial.date
    
    if target == 'volumes':
        financial = financial[['Volume', 'Price']]
        financial['CRSP'] = financial['Price'].str.replace(',', '').astype(float)
        financial['Adj Close'] = financial['Volume']
        
    else:
        financial = financial[['Adj Close', 'Price']]
        financial['CRSP'] = financial['Price'].str.replace(',', '').astype(float)
    
    financial = financial[['Adj Close', 'CRSP']]
    
    financial = financial[financial.index > '2016-01-01']
    new_index = pd.date_range(start=financial.index.min(), end=financial.index.max())
    
    financial = financial.reindex(new_index)
    financial = financial.interpolate()
    
    
    
    financial['StockReturn'] = financial['Adj Close'].diff()
    
    if target == 'volumes':
        financial['MktReturn'] = financial['CRSP'].pct_change()
    else:
        financial['MktReturn'] = financial['CRSP'].diff()
    
    
    financial = financial.dropna()
    
    
    
    
    
    
    
    
    ####################################
    ###########CONSTRUCT AR#############
    #################################### 
    financial['abn_return'] = np.zeros(financial.shape[0])
    for t in range(L1,financial['StockReturn'].shape[0]):
        mu_i = np.mean(financial['StockReturn'][t-L1:t])
        mu_m = np.mean(financial['MktReturn'][t-L1:t])
        
        beta = np.sum((financial['StockReturn'][t-L1:t] - mu_i) - (financial['MktReturn'][t-L1:t] - mu_m)) / np.sum((financial['MktReturn'][t-L1:t] - mu_m)**2)
        alpha = mu_i - beta*mu_m
        abn_return = financial['StockReturn'][t] - alpha - beta*financial['MktReturn'][t]
        
        financial['abn_return'][t] = financial['StockReturn'][t] - alpha - beta*financial['MktReturn'][t]    
    
        
    
    financial['rank_k'] = np.zeros(financial['StockReturn'].shape[0])
    financial['rank_k'][L1:] = financial['abn_return'][L1:].rank()/(1+L1)
    financial['S2_rank'] = np.zeros(financial['StockReturn'].shape[0])
    # for t in range(L1*2,data['StockReturn'].shape[0]-L1):
    #     data['S2_rank'][t] =(1/L1+1)* np.sum(data['rank_k'][L1+t:L1*2+t]) 
    
    financial['S2_rank'] = (1/financial.shape[0]) * np.sum((financial['rank_k'][L1:] - 0.5)**2)
    financial['t_rank'] = (financial['rank_k']-0.5)/np.sqrt(financial['S2_rank'])
    financial['p_val_rank'] = scipy.stats.norm.sf(abs(financial['t_rank']))
    
    alert_dates = (df_results[df_results.alert1 == 1].index).to_list()
    filtered_dates = []
    for date in alert_dates:
        if not filtered_dates or (date - filtered_dates[-1]).days > h_star:
            filtered_dates.append(date)
    
    
    new_df = pd.DataFrame(index=range(h_star*2+1))
    # Iterate over each date in the list
    for date in filtered_dates:
        # Get the 10 days before and after the date, including the date itself
        start_date = date - pd.Timedelta(days=h_star)
        end_date = date + pd.Timedelta(days=h_star)
        
        # Extract the abn_return values for this range
        abn_returns = financial.loc[start_date:end_date, 'abn_return']
        
        # Ensure we have exactly 21 values (10 before, the date itself, 10 after)
        if len(abn_returns) == h_star*2+1:
            # Add this series to the new dataframe as a new column
            new_df[date] = abn_returns.reset_index(drop=True)
        else:
            print(f"Warning: Date {date} does not have a full range of 21 days in the financial data.")
    
    
    
    
    fig, ax = plt.subplots(figsize = (7, 4))
    #ax.plot(np.linspace(-h_star,h_star, h_star*2+1), abnornal_return)
    ax.plot(np.linspace(-h_star,h_star, h_star*2+1), new_df.mean(axis = 1), label = 'AR Average')
    ax.axvline(0, color = 'red')
    ax.axhline(0, linestyle = '-.', linewidth = 2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize = 12, frameon = False)
    fig.savefig('H:/LLongo/Reddit/results/' + ticker + '_AR_train.jpg')
    plt.title(ticker + ', I=' + str(I) + ', Nd=' + str(network_days) + ', r=' + str(red) + '  Train period (2020m12-2022m5)')
    plt.show()
    
    
    #ar1.append(new_df.mean(axis = 1)[11])
    
    ar1.append(new_df)
    
    new_df.mean(axis = 1).to_csv(path_AR + target + "_" + str(h_star) + "_train_" + ticker)
    
    ALERTS[ticker] = df_results
    

    




ti = 0
for tick in Ticker:
    ti = ti + 1
    if ti == 1:
        df_AR = pd.read_csv(path_AR + target + "_" + str(h_star) + "_train_" + tick, index_col = 0)
    else:
        dff = pd.read_csv(path_AR + target + "_" + str(h_star) + "_train_" + tick, index_col = 0)
        df_AR = pd.concat([df_AR, dff], axis = 1)
    

#AR AVERAGE:
fig, ax = plt.subplots(figsize = (7, 4))
#ax.plot(np.linspace(-h_star,h_star, h_star*2+1), abnornal_return)
ax.plot(np.linspace(-h_star,h_star, h_star*2+1), df_AR.mean(axis = 1), label = 'AR Average', marker = 's')
ax.axvline(0, color = 'red')
ax.axhline(0, linestyle = '-.', linewidth = 2)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(fontsize = 12, frameon = False)
plt.title('I=' + str(I) + ', Nd=' + str(network_days) + ', r=' + str(red) + '  Train period (2020m12-2022m5)') 
#fig.savefig('H:/LLongo/Reddit/results/AR_train.jpg')
plt.show()


#CAR AVERAGE:
fig, ax = plt.subplots(figsize = (7, 4))
#ax.plot(np.linspace(-h_star,h_star, h_star*2+1), abnornal_return)
ax.plot(np.linspace(-h_star,h_star, h_star*2+1), df_AR.mean(axis = 1).cumsum(), label = 'CAR Average', marker = 's')
ax.axvline(0, color = 'red')
#ax.axhline(0, linestyle = '-.', linewidth = 2)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend(fontsize = 12, frameon = False)
plt.title('I=' + str(I) + ', Nd=' + str(network_days) + ', r=' + str(red) + '  Train period (2020m12-2022m5)') 
#fig.savefig('H:/LLongo/Reddit/results/CAR_train.jpg')
plt.show()


        
