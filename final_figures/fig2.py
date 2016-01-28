import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy.stats as stats
import matplotlib.lines as mlines

from random_graph.binary_undirected import pure_geometric
from extract.brain_graph import binary_directed
import brain_constants as bc

import in_out_plot_config as cf
import color_scheme

cf.FIGSIZE = (7.5, 3.75)
cf.FONTSIZE = 13

Ls = [0.125, 0.5, 1.5]

fig,axs = plt.subplots(1,2,figsize=cf.FIGSIZE,facecolor='w')

slopes = []
rs = []
i=0
j=0



rs = []
slopes = []
alpha=0.25
markersize=30.
leg = []
cyan = (0.5,0.5,1.0)
cols = ['cyan',(0.2,0.6,1.0),(0.,0.2,0.9)]
deg_bins = np.linspace(0,150,50)

Gbrain,_,_ = binary_directed()
deg_brain = Gbrain.to_undirected().degree().values()

N = 100

axs[0].hist(deg_brain,deg_bins,facecolor=color_scheme.ATLAS,alpha=0.75,normed=True)
leg.append(mlines.Line2D([],[],color=color_scheme.ATLAS,linestyle='-',markersize=13,\
                         label='Connectome',lw=3,alpha=0.75))

for i,L in enumerate(Ls):
    degs=[]
    for k in range(N):
        G,_,_ = pure_geometric(N=bc.num_brain_nodes,N_edges=bc.num_brain_edges_undirected,
                           L=L)
        G_un = G.to_undirected()
        cc_dict = nx.clustering(G_un)
        deg_dict = G_un.degree()
        cc = [cc_dict[k] for k in range(bc.num_brain_nodes)]
        deg = [deg_dict[k] for k in range(bc.num_brain_nodes)]
        degs.extend(deg)

    [slope,intercept,r,_,_] = stats.linregress(deg,cc)
    slopes.append(slope)
    rs.append(r)


    axs[1].scatter(deg,cc,color=cols[i],marker='o',\
                   s=markersize,alpha=alpha,lw=0)

    binned_degs,_ = np.histogram(degs,deg_bins,normed=True)
    dx = deg_bins[1]-deg_bins[0]
    axs[0].plot(deg_bins[:-1]+dx/2.,binned_degs,lw=2,color=cols[i])

    leg.append(mlines.Line2D([],[],color=cols[i],linestyle='-',markersize=13,\
                                    label='L='+str(L),lw=3,alpha=0.7))

xtks = [0,50,100,150]
ytks= [0.,0.25,0.5,0.75,1.0]

axs[0].set_xticks(xtks)
axs[0].set_xlim([0,150])
axs[0].set_ylim([0,0.1])
axs[0].set_yticks([0,0.025,0.05,0.075,0.1])
axs[0].set_xlabel('Degree',fontsize=cf.FONTSIZE)
axs[0].set_ylabel('P(k)',fontsize=cf.FONTSIZE)
axs[0].text(.035*150,0.0915,'a',color='k',fontsize=cf.FONTSIZE,fontweight='bold')

axs[1].set_ylim([0,1.0])
axs[1].set_xlim([0,150])
axs[1].set_xticks(xtks)
axs[1].set_yticks(ytks)
axs[1].set_xlabel('Degree',fontsize=cf.FONTSIZE)
axs[1].set_ylabel('Clustering coefficient',fontsize=cf.FONTSIZE)
axs[1].text(.035*150,0.915,'b',color='k',fontsize=cf.FONTSIZE,fontweight='bold')

ticks = []
for k in range(2):
    ticks += axs[k].get_xticklabels() + axs[k].get_yticklabels()
for tick in ticks:
    tick.set_fontsize(cf.FONTSIZE)

myleg=axs[0].legend(handles=leg,prop={'size':10})

fig.set_tight_layout(True)
plt.draw()
fig.savefig('/mnt/hdd/Documents/dbw/Figures/geometric_cc_vs_deg5.png',dpi=300)


