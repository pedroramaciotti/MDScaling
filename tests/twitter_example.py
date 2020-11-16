import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sn

import mdscaling as mds


# Loading Directed Bipartite Twitter graphs
DG={}
for country in ['chile','france']:
    DG[country] = mds.DiBipartite('datasets/twitter_%s.csv'%country)

# Computing Correspondance Analysis embedding
for country in ['chile','france']:
    DG[country].CA()

# Plotting embeddings for countries
custom_legend=[Line2D([0], [0], color='red', marker='+', lw=0,alpha=1.0,  label='MPs'),
              Line2D([0], [0], color='deepskyblue', lw=8,alpha=0.6,  label='Followers'),]
for country in ['chile','france']:
    g = sn.jointplot(x=1,y=0, data=DG[country].embedding[DG[country].embedding.index.isin(DG[country].bottom_nodes_list)], space=0, color="deepskyblue",kind='hex',ratio=10)
    cbar_ax = g.fig.add_axes()  # x, y, width, height
    cbar_ax = g.fig.add_axes([0.15, .4, .05, .5])  # x, y, width, height
    plt.colorbar(cax=cbar_ax)
    # top
    g.ax_joint.plot(DG[country].embedding[DG[country].embedding.index.isin(DG[country].top_nodes_list)][1],
                    DG[country].embedding[DG[country].embedding.index.isin(DG[country].top_nodes_list)][0],
                    '+',color='r',mew=1.0,ms=7)
    g.ax_joint.legend(handles=custom_legend,loc='lower right',fontsize=12)
    g.ax_joint.set_xlabel('PC2')
    g.ax_joint.set_ylabel('PC1')
    g.ax_joint.set_title('Twitter (%s)'%country,fontsize=14)
    g.ax_joint.set_xlim((-3,3))
    g.ax_joint.set_ylim((-3,3))
    plt.tight_layout()
    plt.savefig('datasets/twitter_%s.pdf'%country)
    plt.savefig('datasets/twitter_%s.png'%country)
    plt.clf()
    plt.close()







