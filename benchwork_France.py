import os
import pandas as pd
import numpy as np
import prince

###########################
# Configuring Tweeter API #
###########################



########################################3

MPs_df = pd.read_csv('local_data/FranceMPsData/MPs.csv')
MPs_df=MPs_df[~MPs_df.twitter_id.isna()]
MPs_df=MPs_df[~MPs_df.twitter_id.duplicated()]

files = os.listdir('local_data/France2019TwitterMPsFollowers/followers_20190522/')

# how many are there in MPs_df ?
MPs_df.Twitter.isin([name.split('.txt')[0] for name in files]).sum() # 1094 out of 1120
foll_files_of_MPs=[f for f in files if f.split('.txt')[0] in MPs_df.Twitter.values] # 1094 out of 1491


# loading followers df
# followers= {}

MPs_followers = pd.DataFrame(columns=['twitter_id','follower_id'])

for f in foll_files_of_MPs:
    twitter_name = f.split('.txt')[0]
    aux_df=pd.DataFrame(columns=MPs_followers.columns)
    # followers[twitter_name] = pd.read_csv('local_data/France2019TwitterMPsFollowers/followers_20190522/'+f,header=None)
    aux_df['follower_id'] = pd.read_csv('local_data/France2019TwitterMPsFollowers/followers_20190522/'+f,header=None)[0].values
    aux_df['twitter_id'] = MPs_df[MPs_df.Twitter==twitter_name].twitter_id if
    aise