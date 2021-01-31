import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from adjustText import adjust_text

def volcano(adata,
            group1,
            group2,
            sig_lvl=3.,
            metric_lvl=3.,
            annotate_gmv=None,
            s=10,
            fontsize=10,
            textsize=8,
            title=False,
            save=False):
    """
    Plot Differential GMV results.
    Please run the Bayesian differential acitvity method of VEGA before plotting ("model.differential_activity()")
    Args:
        adata (Anndata): Scanpy single-cell object.
        group1 (str): Name of reference group.
        group2 (str): Name of out-group.
        sig_lvl (float): Absolute Bayes Factor cutoff. (>=0)
        metric_lvl (float): Mean Absolute Difference cutoff. (>=0)
        annotate_gmv (list): GMV to be displayed. If None, all GMVs passing significance thresholds are displayed.
        s (float): dot size
        fontsize (int): text size for axis
        textsize (int): text size for GMV name display
        title (str): title for plot
        save (str): path to save figure as pdf
    """
    # Check if Anndata is setup correctly
    if '_vega' not in adata.uns.keys():
        raise ValueError('Anndata not setup. Please setup the Anndata object and train the VEGA model.')
    if 'differential' not in adata.uns['_vega'].keys():
        raise ValueError('No differential activity results found in Anndata. Please run model.differential_activity()')
    # Check if group exists
    key_comp = group1 + ' vs.' + group2
    if key_comp not in adata.uns['_vega']['differential'].keys():
        raise ValueError('Group(s) not found. Available comparisons:{}'.format(list(adata.uns['_vega']['differential'].keys())))
    
    dfe_res = adata.uns['_vega']['differential'][key_comp]

    xlim_v = np.abs(dfe_res['bayes_factor']).max() + 0.5
    ylim_v = dfe_res['mad'].max()+0.5

    idx_sig = np.arange(len(dfe_res['bayes_factor']))[(np.abs(dfe_res['bayes_factor'])>sig_lvl) & (np.abs(dfe_res['mad'])>metric_lvl)]
    plt.scatter(dfe_res['bayes_factor'], dfe_res['mad'], color='k', s=s, alpha=0.7, linewidth=0)
    plt.scatter(dfe_res['bayes_factor'][idx_sig], dfe_res['mad'][idx_sig], color='red', s=s*2, linewidth=0)
    plt.vlines(x=-sig_lvl, ymin=-0.5, ymax=ylim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    plt.vlines(x=sig_lvl, ymin=-0.5, ymax=ylim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    plt.hlines(y=metric_lvl, xmin=-xlim_v, xmax=xlim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    texts = []
    if not annotate_gmv:
        for i in idx_sig:
            name = adata.uns['_vega']['gmv_names'][i]
            x = dfe_res['bayes_factor'][i]
            y = dfe_res['mad'][i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size':textsize}))
    else:
        for name in annotate_gmv:
            i = adata.uns['_vega']['gmv_names'].index(name)
            x = dfe_res['bayes_factor'][i]
            y = dfe_res['mad'][i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size':textsize}))
        

    plt.xlabel(r'$\log_e$(Bayes factor)', fontsize=fontsize)
    plt.ylabel('MD', fontsize=fontsize)
    plt.ylim([0,ylim_v])
    plt.xlim([-xlim_v,xlim_v])
    if title:
        plt.title(title+'(|K|>%.1f)'%(sig_lvl))
    adjust_text(texts, only_move={'texts':'xy'}, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    if save:
        plt.savefig(save, format='pdf', dpi=150, bbox_inches='tight')
    plt.show()


def gmv_embedding(adata, x, y, color=None, palette=None, title=None, save=False, **sct_kwds):
    """ 
    2-D scatter plot in GMV space.
    Args:
        adata (Anndata): Scanpy single-cell object. VEGA analysis needs to be run before.
        x (str): GMV name for x-coordinates (eg. 'REACTOME_INTERFERON_SIGNALING').
        y (str): GMV name for y-coordinates (eg. 'REACTOME_INTERFERON_SIGNALING').
        color (str): Categorical field of Anndata.obs to color single-cells.
        title (str): Plot title.
        save (str): Path to save plot.
        sct_kwds: kwargs for matplotlib.pyplot.scatter function.
    """
    if 'X_vega' not in adata.obsm.keys():
        raise ValueError("No GMV coordinates found in Anndata. Run 'adata.obsm['X_vega'] = model.to_latent()'")
    # Check if dim exist
    if not all([_check_exist(adata, x), _check_exist(adata, y), _check_exist(adata, color)]):
        raise ValueError("At least one of passed (x, y, color) names not found in Anndata. Make sure those names exist.")
    x_i = adata.uns['_vega']['gmv_names'].index(x)
    y_i = adata.uns['_vega']['gmv_names'].index(y)
    dim1 = adata.obsm['X_vega'][:,x_i]
    dim2 = adata.obsm['X_vega'][:,y_i]
    color_val = _get_color_values(adata, color, palette)
    plt.scatter(x=dim1, y=dim2, c=color_val, **sct_kwds)
    plt.xlabel(x)
    plt.ylabel(y)
    if title:
        plt.title(title)
    if save:
        plt.savefig(save, format='pdf', dpi=150, bbox_inches='tight')
    plt.show()

#def gmv_dotplot():
    #return

def _check_exist(adata, x):
    """ Check if dimension exist in Anndata. """
    if (x not in list(adata.obs)) or (x not in list(adata.var)) or (x not in adata.uns['_vega']['gmv_names']):
        exist = False
    else:
        exist = True
    return exist

def _get_color_values(adata, var, palette):
    """ Value to color. TODO: Add support for gene variable."""
    if (not var) or (var not in adata.uns['_vega']['gmv_names'].keys()) or (var not in list(adata.obs)):
        return "lightgray"
    elif var in adata.uns['_vega']['gmv_names'].keys():
        if not palette:
            palette = 'viridis'
        cmap = mpl.cm.get_cmap(palette)
        val_vec = adata.obsm['X_vega'][:,adata.uns['_vega']['gmv_names'].index(var)]
        color_vec = cmap(val_vec)
        return color_vec
    else:
        if adata.obs[var].dtype == 'category':
            if not palette:
                palette = 'deep'
            lbl = adata.obs[var].unique()
            n = len(lbl)
            cval = sns.color_palette(palette, n)
            color_map = dict(zip(lbl, cval))
            color_vec = adata.obs[var].map(color_map)
            return color_vec
        else:
            if not palette:
                palette = 'viridis'
            cmap = mpl.cm.get_cmap(palette)
            val_vec = adata.obs[var]
            color_vec = cmap(val_vec)
            return color_vec
