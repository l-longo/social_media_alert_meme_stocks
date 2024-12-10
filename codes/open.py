import pandas as pd

path_X = 'your_path' #select your path.
path_save = 'your_path'

#OPEN COMMENTS AND SUBMISSIONS:
df__comments = pd.read_csv(path_X + '\wallstreetbets__comments.csv')
df__comments.index = pd.to_datetime(df__comments['created_utc'], unit = 's').dt.date
df__comments.index = pd.to_datetime(df__comments.index, format= '%d/%m/%Y')

df_comments = pd.read_csv(path_X + '\wallstreetbets_comments.csv')
df_comments.index = pd.to_datetime(df_comments['created_utc'], unit = 's').dt.date
df_comments.index = pd.to_datetime(df_comments.index, format= '%d/%m/%Y')

df_comments2 = pd.read_csv(path_X + '\wallstreetbets2_comments.csv')
df_comments2.index = pd.to_datetime(df_comments2['created_utc'], unit = 's').dt.date
df_comments2.index = pd.to_datetime(df_comments2.index, format= '%d/%m/%Y')


df__submissions = pd.read_csv(path_X + '\wallstreetbets__submissions.csv')
df__submissions.index = pd.to_datetime(df__submissions['created_utc'], unit = 's').dt.date
df__submissions.index = pd.to_datetime(df__submissions.index, format= '%d/%m/%Y')

df_submissions = pd.read_csv(path_X + '\wallstreetbets_submissions.csv')
df_submissions.index = pd.to_datetime(df_submissions['created_utc'], unit = 's').dt.date
df_submissions.index = pd.to_datetime(df_submissions.index, format= '%d/%m/%Y')

df_submissions2 = pd.read_csv(path_X + '\wallstreetbets2_submissions.csv')
df_submissions2.index = pd.to_datetime(df_submissions2['created_utc'], unit = 's').dt.date
df_submissions2.index = pd.to_datetime(df_submissions2.index, format= '%d/%m/%Y')


df__submissions.rename(columns={'id': 'id_submission'}, inplace=True)
df_submissions.rename(columns={'id': 'id_submission'}, inplace=True)
df_submissions2.rename(columns={'id': 'id_submission'}, inplace=True)


#MERGE COMMENTS AND SUBMISSIONS BASED ON THE ID:
df__submissions['parent_id'] = 't3_' + df__submissions['id_submission']
df_submissions['parent_id'] = 't3_' + df_submissions['id_submission']
df_submissions2['parent_id'] = 't3_' + df_submissions2['id_submission']

df_subm_merged = pd.concat([df__submissions, df_submissions, df_submissions2])
df_comm_merged = pd.concat([df__comments, df_comments, df_comments2])

df_subm_merged['utc'] = df_subm_merged.created_utc
df_comm_merged['utc'] = df_comm_merged.created_utc

df_merged = pd.merge(df_comm_merged, df_subm_merged, on='utc', how='left')

df_merged.index = pd.to_datetime( df_merged.created_utc_x.fillna(df_merged['created_utc_y']), unit = 's').dt.date
df_merged.index = pd.to_datetime(df_merged.index, format= '%d/%m/%Y')

df_merged['title'] = df_merged.title.fillna(method='ffill')





tickers = ['amzn', 'googl', 'meta', 'pypl', 'nflx', 'cmcsa', 'pep', 'csco', 
          'intc', 'cost', 'qcom', 'amgn', 'sbux']
tickers = ['aal', 'dis', 'htz', 'djt', 'tlry']

df_merged['title'] = df_merged.title.str.lower()
df_subm_merged['title'] = df_subm_merged.title.str.lower()
filtered_comments = df_comm_merged[df_comm_merged['parent_id'].isin(df_subm_merged['parent_id'])]


for ticker in tickers:
    print(ticker)
    df_ticker = df_merged[df_merged['title'].fillna(method='bfill').str.contains(ticker)]
    df_ticker.to_csv('X:/Reddit/qf_review/reddit_data/' +  ticker + '_reddit.csv')
    
    df_subm_ticker = df_subm_merged[df_subm_merged['title'].fillna(method='bfill').str.contains(ticker)]
    df_subm_ticker.to_csv('X:/Reddit/qf_review/reddit_data/' +  ticker + '_subm_reddit.csv')


df_subm_merged.to_csv('path_save/total_subm.csv')
