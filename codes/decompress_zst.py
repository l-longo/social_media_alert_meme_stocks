import zstandard
import pandas as pd

path_X = 'your_path'

import os
os.chdir(path_X)  
#comments
!python to_csv.py wallstreetbets_comments.zst wallstreetbets_comments.csv body,created_utc,parent_id,distinguished,edited,id,score,author
!python to_csv.py wallstreetbets__comments.zst wallstreetbets__comments.csv body,created_utc,parent_id,distinguished,edited,id,score,author
!python to_csv.py wallstreetbets2_comments.zst wallstreetbets2_comments.csv body,created_utc,parent_id,distinguished,edited,id,score,author
#submissions
!python to_csv.py wallstreetbets_submissions.zst wallstreetbets_submissions.csv created_utc,distinguished,edited,id,score,title,url,num_comments,link_flair_text,over_18,is_self,permalink,selftext,author
!python to_csv.py wallstreetbets__submissions.zst wallstreetbets__submissions.csv created_utc,distinguished,edited,id,score,title,url,num_comments,link_flair_text,over_18,is_self,permalink,selftext,author
!python to_csv.py wallstreetbets2_submissions.zst wallstreetbets2_submissions.csv created_utc,distinguished,edited,id,score,title,url,num_comments,link_flair_text,over_18,is_self,permalink,selftext,author
