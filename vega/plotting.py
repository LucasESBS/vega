import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib as mpl
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np
import pandas as pd
from scanpy.plotting import embedding
from scanpy import settings
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
            figsize = None,
            title=False,
            save=False):
    """
    Plot Differential GMV results.
    Please run the Bayesian differential acitvity method of VEGA before plotting ("model.differential_activity()")
    
    Parameters
    ----------
    adata
        scanpy single-cell object
    group1
        name of reference group
    group2
        name of out-group
    sig_lvl
        absolute Bayes Factor cutoff (>=0)
    metric_lvl
        mean Absolute Difference cutoff (>=0)
    annotate_gmv
        GMV to be displayed. If None, all GMVs passing significance thresholds are displayed
    s
        dot size
    fontsize
        text size for axis
    textsize
        text size for GMV name display
    title
        title for plot
    save
        path to save figure as pdf
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
    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(dfe_res['bayes_factor'], dfe_res['mad'],
                 color='grey', s=s, alpha=0.8, linewidth=0,
                 rasterized=settings._vector_friendly)
    ax.scatter(dfe_res['bayes_factor'][idx_sig], dfe_res['mad'][idx_sig],
                 color='red', s=s*2, linewidth=0,
                 rasterized=settings._vector_friendly)
    ax.vlines(x=-sig_lvl, ymin=-0.5, ymax=ylim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    ax.vlines(x=sig_lvl, ymin=-0.5, ymax=ylim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    ax.hlines(y=metric_lvl, xmin=-xlim_v, xmax=xlim_v, color='black', linestyles='--', linewidth=1., alpha=0.2)
    texts = []
    if not annotate_gmv:
        for i in idx_sig:
            name = adata.uns['_vega']['gmv_names'][i]
            x = dfe_res['bayes_factor'][i]
            y = dfe_res['mad'][i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size':textsize}))
    else:
        for name in annotate_gmv:
            i = list(adata.uns['_vega']['gmv_names']).index(name)
            x = dfe_res['bayes_factor'][i]
            y = dfe_res['mad'][i]
            texts.append(plt.text(x=x, y=y, s=name, fontdict={'size':textsize}))
        

    ax.set_xlabel(r'$\log_e$(Bayes factor)', fontsize=fontsize)
    ax.set_ylabel('MD', fontsize=fontsize)
    ax.set_ylim([0,ylim_v])
    ax.set_xlim([-xlim_v,xlim_v])
    if title:
        ax.set_title(title, fontsize=fontsize)
    adjust_text(texts, only_move={'texts':'xy'}, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    plt.grid(False)
    if save:
        plt.savefig(save, format=save.split('.')[-1], dpi=rcParams['savefig.dpi'], bbox_inches='tight')
    plt.show()


def gmv_embedding(adata, x, y, color=None, palette=None, title=None, save=False, sct_kwds=None):
    """ 
    2-D scatter plot in GMV space.

    Parameters
    ----------
    adata
        scanpy single-cell object. VEGA analysis needs to be run before
    x
        GMV name for x-coordinates (eg. 'REACTOME_INTERFERON_SIGNALING')
    y
        GMV name for y-coordinates (eg. 'REACTOME_INTERFERON_SIGNALING')
    color
        categorical field of Anndata.obs to color single-cells
    title
        plot title
    save
        path to save plot
    sct_kwds
        kwargs for matplotlib.pyplot.scatter function
    """
    if 'X_vega' not in adata.obsm.keys():
        raise ValueError("No GMV coordinates found in Anndata. Run 'adata.obsm['X_vega'] = model.to_latent()'")
    # Check if dim exist
    if not color:
        if not all([_check_exist(adata, x), _check_exist(adata, y)]):
            raise ValueError("At least one of passed (x, y) names not found in Anndata. Make sure those names exist.")
    else:
       if not all([_check_exist(adata, x), _check_exist(adata, y), _check_exist(adata, color)]):
        raise ValueError("At least one of passed (x, y, color) names not found in Anndata. Make sure those names exist.") 
    x_i = list(adata.uns['_vega']['gmv_names']).index(x)
    y_i = list(adata.uns['_vega']['gmv_names']).index(y)
    dim1 = adata.obsm['X_vega'][:,x_i]
    dim2 = adata.obsm['X_vega'][:,y_i]
    color_val = _get_color_values(adata, color, palette)
    sct_kwds = {} if sct_kwds is None else sct_kwds.copy()
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
    if (x in list(adata.obs)) or (x in list(adata.var)) or (x in adata.uns['_vega']['gmv_names']):
        exist = True
    else:
        exist = False
    return exist

def _get_color_values(adata, var, palette):
    """ Value to color. TODO: Add support for gene variable."""
    if (not var) and (var not in adata.uns['_vega']['gmv_names']) and (var not in list(adata.obs)):
        return "lightgray"
    elif var in list(adata.uns['_vega']['gmv_names']):
        if not palette:
            palette = 'viridis'
        cmap = mpl.cm.get_cmap(palette)
        val_vec = adata.obsm['X_vega'][:,list(adata.uns['_vega']['gmv_names']).index(var)]
        color_vec = cmap(val_vec)
        return color_vec
    else:
        if adata.obs[var].dtype == 'category':
            if not palette:
                palette = 'tab10'
            lbl = adata.obs[var].unique()
            n = len(lbl)
            cval = sns.color_palette(palette, n)
            color_map = dict(zip(lbl, cval))
            color_vec = [color_map[k] for k in adata.obs[var]]
            return color_vec
        else:
            if not palette:
                palette = 'viridis'
            cmap = mpl.cm.get_cmap(palette)
            val_vec = adata.obs[var]
            color_vec = cmap(val_vec)
            return color_vec


def gmv_plot(adata, x, y, color=None, title=None, palette=None):
    """
    GMV embedding plot, but using the Scanpy plotting API.
    
    Parameters
    ----------
    adata
        scanpy single-cell dataset
    x
        GMV name for x-coordinates (eg. 'REACTOME_INTERFERON_SIGNALING')
    y
        GMV name for x-coordinates (eg. 'REACTOME_INTERFERON_SIGNALING')
    color
        .obs field to color by
    title
        title for the plot
    palette
        matplotlib colormap to be used
    """
    if 'X_vega' not in adata.obsm.keys():
        raise ValueError("No GMV coordinates found in Anndata. Run 'adata.obsm['X_vega'] = model.to_latent()'")
    # Check if dim exist
    if not color:
        if not all([_check_exist(adata, x), _check_exist(adata, y)]):
            raise ValueError("At least one of passed (x, y) names not found in Anndata. Make sure those names exist.")
    else:
       if not all([_check_exist(adata, x), _check_exist(adata, y), _check_exist(adata, color)]):
        raise ValueError("At least one of passed (x, y, color) names not found in Anndata. Make sure those names exist.")
    # Components are indexed starting at 1 - so add 1 to indices
    x_i = list(adata.uns['_vega']['gmv_names']).index(x)+1
    y_i = list(adata.uns['_vega']['gmv_names']).index(y)+1
    # Call Scanpy embedding wrapper
    fig = embedding(adata,
                    basis='X_vega',
                    color=color,
                    components=[x_i, y_i],
                    title=title,
                    palette=palette,
                    return_fig=True,
                    show=False).gca()
    fig.set_xlabel(x)
    fig.set_ylabel(y)
    plt.show()
    return

def loss(model, plot_validation=True):
    """
    Plot training loss and validation if plot_validation is True. 
    """
    train_hist = model.epoch_history['train_loss']
    n_epochs = len(train_hist)
    plt.plot(np.arange(n_epochs), train_hist, label='Training loss', color='blue')
    if plot_validation:
        plt.plot(np.arange(n_epochs),
                 model.epoch_history['valid_loss'],
                 label='Validation loss',
                 color='orange'
                )
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    return

def rank_gene_weights(model, gmv_list, n_genes=10, color_in_set=True, n_panels_per_row=3, fontsize=8, star_names=[], save=False):
    """
    Plot gene members of input GMVs according to their magnitude (abs(w)).
    Inspired by scanpy.pl.rank_gene_groups() API.

    Parameters:
    -----------
    model
        VEGA trained model
    gmv_list
        list of GMV names
    n_genes
        number of top gene to display
    color_in_set
        Whether to color genes annotated as part of GMVs differently.
    n_panels_per_row
        number of panels max. per row
    star_names
        Name of genes to be highlighted with stars
    save
        path to save figure
    """
    if not model.is_trained_:
        raise ValueError('Model is not trained. Please train the model before.')
    w = model.decoder._get_weights().data
    gmv_names = list(model.adata.uns['_vega']['gmv_names'])
    gene_names = model.adata.var.index.tolist()
    
    n_panelx = min(n_panels_per_row, len(gmv_list))
    n_panely = np.ceil(len(gmv_list) / n_panelx).astype(int)
    
    from matplotlib import gridspec
    fig = plt.figure(
        figsize=(
            n_panelx * rcParams['figure.figsize'][0],
            n_panely * rcParams['figure.figsize'][1],
        )
    )
    gs = gridspec.GridSpec(nrows=n_panely, ncols=n_panelx, wspace=0.22, hspace=0.3)
    ax0 = None
    ymin = np.Inf
    ymax = -np.Inf
    
    
    for l, k in enumerate(gmv_list):
        # Get values
        i = gmv_names.index(k)
        w_i = w[:,i].detach().numpy()
        sort_idx = np.argsort(np.abs(w_i))[::-1]
        abs_w = np.abs(w_i)[sort_idx][:n_genes]
        genes = np.array(gene_names)[sort_idx][:n_genes]
        
        # Set plot params
        ymin = np.min(abs_w)
        ymax = np.max(abs_w)
        ymax += 0.3*(ymax - ymin)

        ax = fig.add_subplot(gs[l])
        ax.set_ylim(ymin, ymax)

        ax.set_xlim(-0.9, n_genes - 0.1)
        for ig, gene_name in enumerate(genes):
            if color_in_set:
                in_set = bool(model.adata.uns['_vega']['mask'][sort_idx[ig],i])
                col = 'black' if in_set else 'red'
            else:
                col = 'black'
            gene_name += '*' if gene_name in star_names else ''
            ax.text(
                ig,
                abs_w[ig],
                gene_name,
                rotation='vertical',
                verticalalignment='bottom',
                horizontalalignment='center',
                fontsize=fontsize,
                color=col
            )
        ax.set_title('{}'.format(k))
        if l >= n_panelx * (n_panely - 1):
            ax.set_xlabel('ranking')
        if l % n_panelx == 0:
            ax.set_ylabel('Weight magnitude')
    if color_in_set:
        leg = [Line2D([0], [0], marker='o', color='w', label='In set',
                          markerfacecolor='black', markersize=5),
                    Line2D([0], [0], marker='o', color='w', label='Not in set',
                          markerfacecolor='red', markersize=5)]
        plt.legend(handles=leg, loc='upper right')
    #plt.grid(False)
    if save:
        plt.savefig(save, format=save.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()
    return


def weight_heatmap(model, 
                cluster=True, 
                cmap='viridis', 
                display_gmvs='all', 
                display_genes='all',
                title=None,
                figsize=None,
                save=False,
                hm_kwargs=None):
    """
    Heatmap plots of weights.

    Parameters
    ----------
    cluster
        If True, use hierarchical clustering (seaborn.clustermap)
    cmap
        colormap to use
    display_gmvs
        If all, display all latent variables weights. Else (list) only subset.
    """
    if cluster:
        fn = sns.clustermap
    else:
        fn = sns.heatmap
    w = model.decoder._get_weights().data.numpy()
    gmv_names = model.adata.uns['_vega']['gmv_names']
    gene_names = model.adata.var_names
    df = pd.DataFrame(data=w, index=gene_names, columns=gmv_names)
    if display_gmvs != 'all' and type(display_gmvs)==list:
        df = df[display_gmvs]
    if display_genes != 'all' and type(display_genes)==list:
        df = df.loc[display_genes]
    hm_kwargs = {} if hm_kwargs is None else hm_kwargs
    if figsize:
        fig = plt.figure(figsize)
    ax = fn(df.T, cmap=cmap, **hm_kwargs, cbar_kws={'label': 'Weight magnitude'})
    ax.set_xlabel('Genes')
    if title:
        plt.title(title)
    if save:
        print('Saving figure at %s'%(save))
        plt.savefig(save, format=save.split('.')[-1], dpi=300, bbox_inches='tight')
    plt.show()
    return   
    


def _make_pretty(string):
    """ Make GMV name pretty """
    if 'UNANNOTATED' in string:
        s = '_'.join(string.split('_')).lower()
    else:
        s = ' '.join(string.split('_')[1:]).lower() + ' activity'
    return s.capitalize()
