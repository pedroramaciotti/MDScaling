import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr
from scipy.stats import spearmanr

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

def corr_sig(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = spearmanr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix

# MPs, marpor, ches
MPs_df = pd.read_csv('local_data/FranceMPsData/MPs_PC_marpor_CHES.csv')
MPs_df.rename(columns={'2':'PC3'},inplace=True)
marpor_cols = [c for c in MPs_df.columns if (c.startswith('marpor') and c!='marpor_partyname')]
ches_cols = [c for c in MPs_df.columns if (c.startswith('ches') and c!='ches_party')]

# MARPOR
corr = MPs_df[['PC1','PC2','PC3']+marpor_cols].corr(method='spearman').iloc[:3,3:]
# mask = np.triu(np.ones_like(corr, dtype=np.bool))
mask=corr.isna().values
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig = plt.figure(figsize=(70,2.5))# width, height inches
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
# hm.set(xticks=np.arange(len(marpor_cols))+0.5)
# hm.set(xticklabels=marpor_cols)
plt.tight_layout()
plt.savefig('local_figures/AngloGJ/MPs_marpor_twitter_spearman.pdf')

#
selected_marpor_variables=['marpor_rile','marpor_planeco','marpor_markeco']
g = sns.pairplot(MPs_df[['PC%d'%(i+1) for i in range(3)]+selected_marpor_variables],kind="reg")
g.map(corrfunc)
plt.savefig('local_figures/AngloGJ/MPs_marpor_rile_pairplot.pdf')

# CHES
corr = MPs_df[['PC1','PC2','PC3']+ches_cols].corr(method='spearman').iloc[:3,3:]
# mask = np.triu(np.ones_like(corr, dtype=np.bool))
mask=np.invert(np.tril(corr_sig(MPs_df[['PC1','PC2','PC3']+ches_cols])[:3,3:]<0.5))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
fig = plt.figure(figsize=(30,4))# width, height inches
ax = fig.add_subplot(1,1,1)
hm=sns.heatmap(corr, cmap=cmap, 
                center=0,
                # square=True, 
                # mask=mask,
                linewidths=.1, 
                # cbar_kws={"shrink": .5},
                annot=True,
                fmt='.2f',
                ax=ax
                )
# hm.set(xticks=np.arange(len(ches_cols))+0.5)
# hm.set(xticklabels=ches_cols)
plt.tight_layout()
plt.savefig('local_figures/AngloGJ/MPs_ches_twitter_spearman.pdf')

