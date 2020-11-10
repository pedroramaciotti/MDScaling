import os
import pandas as pd
import numpy as np
import prince
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from scipy.stats import spearmanr

from statsmodels.regression.linear_model import OLS  
import statsmodels.api as sm  

    
def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    (rs, ps) = spearmanr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f} ".format(r),
                xy=(.1, .9), xycoords=ax.transAxes)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.4, .9), xycoords=ax.transAxes)
    ax.annotate("r_s = {:.2f} ".format(rs),
                xy=(.1, .8), xycoords=ax.transAxes)
    ax.annotate("p_s = {:.3f}".format(ps),
                xy=(.4, .8), xycoords=ax.transAxes)


N_dims = 2

# LOADING DATA
########################################

MPs_df = pd.read_csv('local_data/ChileMPsData/ChileanMPs.csv')
MPs_followers=pd.read_csv('local_data/Chile2020TwitterMPsFollowers/ChileanMPsFollowers.csv')
MARPOR=pd.read_csv('local_data/MARPOR_LA/ChileanTwitterMarpor2019Data.csv')

MARPOR.drop(labels='Unnamed: 0',inplace=True,axis=1)
MARPOR.drop(labels='twitter_affiliation',inplace=True,axis=1)
MARPOR.drop_duplicates(inplace=True)
raise

# Preprocessing
########################################

# delete followers that are also MP
MPs_followers=MPs_followers[~MPs_followers.follower_id.isin(MPs_df.twitter_id.values)]

# delete MPs that are not in the table with info
MPs_followers=MPs_followers[MPs_followers.twitter_id.isin(MPs_df.twitter_id.values)]

# drop duplicates
MPs_followers.drop_duplicates(inplace=True)

# 
simple_marpor_cols = MARPOR.columns[-5:]
complex_marpor_cols = MARPOR.columns[1:-5]

# Populating with MARPOR data
########################################


marpor_cols = MARPOR.columns

for c in MARPOR.columns:
    MPs_df[c] = MPs_df['marpor_affiliation'].map(pd.Series(index=MARPOR['marpor_affiliation'].values,data=MARPOR[c].values))


# MDS
########################################

for friends_thrshld in [3]:
    follower_degrees = MPs_followers[['follower_id','twitter_id']].groupby('follower_id').count()['twitter_id'].sort_values(ascending=False)
    follower_degrees=follower_degrees[follower_degrees>=friends_thrshld]
    MPs_followers_f = MPs_followers[MPs_followers.follower_id.isin(follower_degrees.index )]
    
    # matrix
    MPs_followers_f['weights'] = 1  
    M = MPs_followers_f.pivot(index='follower_id',columns='twitter_id',values='weights')
    M.fillna(0,inplace=True)      
    
    # deleting repeated rows
    M.drop_duplicates(inplace=True)
    
    # correspondence analysis
    ca = prince.CA(n_components=N_dims,n_iter=3,copy=True,check_input=True,engine='auto',random_state=42)
    M.columns.rename('MPs', inplace=True)
    M.index.rename('Users', inplace=True)
    ca = ca.fit(M)
    row_coords = ca.row_coordinates(M)
    col_coords = ca.column_coordinates(M)
    ca.explained_inertia_
    
    col_rename={}
    for c in row_coords.columns:
        col_rename[c]='PC%d'%(c+1)
    row_coords.rename(columns=col_rename,inplace=True)
    col_coords.rename(columns=col_rename,inplace=True)
    
    # Populating with MARPOR data
    ########################################
    
    
    for c in MARPOR.columns[1:]:
        col_coords[c] = col_coords.index.map(pd.Series(index=MPs_df.twitter_id.values,data=MPs_df[c].values))
    
    
    # Correlations
    ########################################
      
    corr = col_coords[['PC%d'%(i+1) for i in range(N_dims)]+complex_marpor_cols.tolist()].corr(method='spearman').iloc[:N_dims,N_dims:]
    # mask = np.triu(np.ones_like(corr, dtype=np.bool))
    mask=corr.isna().values
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig = plt.figure(figsize=(34,2.5))# width, height inches
    ax = fig.add_subplot(1,1,1)
    hm=sns.heatmap(corr, cmap=cmap, 
                    center=0,
                    # square=True, 
                    # mask=mask
                    linewidths=.1, 
                    # cbar_kws={"shrink": .5},
                    annot=True,
                    fmt='.2f',
                    ax=ax
                    )
    hm.set(xticks=np.arange(len(complex_marpor_cols))+0.5)
    hm.set(xticklabels=complex_marpor_cols)
    plt.tight_layout()
    plt.savefig('local_figures/Chile/corr_complexvar_heatmap_thrshld%d.pdf'%friends_thrshld)
    
    
    corr = col_coords[['PC%d'%(i+1) for i in range(N_dims)]+simple_marpor_cols.tolist()].corr(method='spearman').iloc[:N_dims,N_dims:]
    # mask = np.triu(np.ones_like(corr, dtype=np.bool))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    fig = plt.figure(figsize=(4,4))# width, height inches
    ax = fig.add_subplot(1,1,1)
    hm=sns.heatmap(corr, cmap=cmap, center=0,
                square=True, 
                linewidths=.1, 
                cbar_kws={"shrink": .85},
                # mask=mask,
                annot=True,
                fmt='.2f',ax=ax)
    hm.set(xticks=np.arange(len(simple_marpor_cols))+0.5)
    hm.set(xticklabels=simple_marpor_cols)
    plt.tight_layout()
    plt.savefig('local_figures/Chile/corr_simplevar_heatmap_thrshld%d.pdf'%friends_thrshld)
    
    
    
    g = sns.pairplot(col_coords[['PC%d'%(i+1) for i in range(N_dims)]+simple_marpor_cols.tolist()],kind="reg")
    g.map(corrfunc)
    plt.savefig('coor_simplevar_pairplot_thrshld%d.png'%friends_thrshld)
    
    
MPs_full_data = MPs_df.copy(deep=True)

MPs_full_data['PC1'] = col_coords['PC1'].values
MPs_full_data['PC2'] = col_coords['PC2'].values

MPs_full_data.to_csv('export_threshold3_gabriel.csv',index=False)

    
# friends_thrshld = 7
# Y=col_coords.dropna()[['PC1','PC2']]
# X=col_coords.dropna()[marpor_cols]
# X = sm.add_constant(X)
# model = sm.OLS(Y,X)    
# results = model.fit()
# results.params




               
               
               
               