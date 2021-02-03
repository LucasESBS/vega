import matplotlib.pyplot as plt
from adjustText import adjust_text
import numpy as np

def volcano_plot(dfe_res, 
                pathway_list,
                sig_lvl=3.,
                metric_lvl=3.,
                to_plot=None,
                metric='mad', 
                figsize=[12,5],
                s=10,
                fontsize=10,
                textsize=8,
                title=False,
                save=False):
    """
    Plot Differential GMV results.
    Args:
        dfe_res (dict): Results of VEGA differential GMV test
        pathway_list (list): List with GMV names
        sig_lvl (float): Absolute Bayes Factor cutoff
        metric_lvl (float): Mean Absolute Difference cutoff
        to_plot (dict): dictionary of {GMV:alias} for labelling particular GMVs. If None, all significant GMVs are displayed.
        metric (str): y-axis metric (MAD)
    kwargs:
        figsize (list): size of figure
        s (float): dot size
        fontsize (int): text size for axis
        textsize (int): text size for GMV name display
        title (str): title for plot
        save (str): path to save figure as pdf
    """ 
    plt.figure(figsize=figsize)
    xlim_v = np.abs(dfe_res['bayes_factor']).max() + 0.5
    ylim_v = dfe_res[metric].max()+0.5

    idx_sig = np.arange(len(dfe_res['bayes_factor']))[(np.abs(dfe_res['bayes_factor'])>sig_lvl) & (np.abs(dfe_res[metric])>lfc_lvl)]
    plt.scatter(dfe_res['bayes_factor'], dfe_res[metric], color='grey', s=s, alpha=0.8, linewidth=0)
    plt.scatter(dfe_res['bayes_factor'][idx_sig], dfe_res[metric][idx_sig], color='red', s=s*2, linewidth=0)
    plt.vlines(x=-sig_lvl, ymin=-0.5, ymax=ylim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    plt.vlines(x=sig_lvl, ymin=-0.5, ymax=ylim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    plt.hlines(y=lfc_lvl, xmin=-xlim_v, xmax=xlim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    texts = []
    if to_plot is None:
        for i in idx_sig:
            name = pathway_list[i]
            x = dfe_res['bayes_factor'][i]
            y = dfe_res[metric][i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size':textsize}))
    else:
        idx_plot = [(pathway_list.index(f),to_plot[f]) for f in to_plot]
        for i in idx_plot:
            name = i[1]
            x = dfe_res['bayes_factor'][i[0]]
            y = dfe_res[metric][i[0]]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size':textsize}))
    
    plt.xlabel(r'$\log_e$(Bayes factor)', fontsize=fontsize)
    if metric == 'mad':
        plt.ylabel('MD', fontsize=fontsize)
    plt.ylim([0,ylim_v])
    plt.xlim([-xlim_v,xlim_v])
    if title:
        plt.title(title+' (|K|>%.1f)'%(sig_lvl))
    adjust_text(texts, only_move={'texts':'xy'}, arrowprops=dict(arrowstyle="-", color='k', lw=2))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if save:
        plt.savefig(save, format='pdf', dpi=150, bbox_inches='tight')
    plt.show()
