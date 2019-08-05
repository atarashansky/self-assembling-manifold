import colorlover as cl
import utilities as ut
import numpy as np
import scipy.sparse as sp
import ipywidgets as widgets
import plotly.graph_objs as go
from SAM import SAM
import pandas as pd

from ipyevents import Event
from ipywidgets import Widget

__version__ = '0.6.6'

class SAMGUI(object):

    def __init__(self,sam, close_all_widgets=False):
        if close_all_widgets:
            Widget.close_all()

        self.sam = sam
        self.sams = [self.sam]
        self.GENE_KEY=''
        self.selected = [np.zeros(self.sam.adata.shape[0],dtype='bool')]
        self.selected[0][:] = True
        self.active_labels = [np.zeros(self.selected[0].size,dtype='int')]
        try:
            self.marker_genes = [np.array(list(sam.adata.var_names))[np.argsort(-sam.adata.var['weights'].get_values())]]
            self.marker_genes_tt = ['Genes ranked by SAM weights.']
        except KeyError:
            self.marker_genes = [np.array(list(sam.adata.var_names))]
            self.marker_genes_tt = ['Unranked genes (SAM not run).']

        self.selected_cells = [np.array(list(self.sam.adata.obs_names))]
        self.ds = [0]

        self.preprocess_args = self.sam.preprocess_args.copy()
        self.preprocess_args_init = self.preprocess_args.copy()

        self.run_args = self.sam.run_args.copy()
        self.run_args_init = self.run_args.copy()

        self.pp_box = self.init_preprocess()
        self.rs_box = self.init_run_sam()
        self.cs_box = self.init_cs()
        self.out = widgets.Output(layout={'border': '1px solid black',
                                          'height':'600px','width':'600px'})
        children = [self.cs_box,self.rs_box,self.pp_box,self.out]
        tab = widgets.Tab()
        tab.children = children
        names = ['Interact','Run SAM','Preprocess data','SAM output']
        for i in range(len(children)):
            tab.set_title(i,names[i])
        self.tab = tab

        self.stab = widgets.Tab()
        self.create_plot(0,'Full dataset')
        self.stab.observe(self.on_switch_tabs,'selected_index')
        items = [self.stab,tab]
        gb = widgets.HBox(items)

        self.SamPlot = gb

    def create_plot(self,i, title):
        projs = list(self.sams[i].adata.obsm.keys())
        if 'X_umap' in projs:
            p = 'X_umap'
        elif len(projs) > 1:
            p = np.array(projs)[np.where(np.array(projs)!='X_pca')[0]]
        else:
            p = 'X_pca'

        if p not in projs:
            xdata = np.zeros(self.sams[i].adata.shape[0])
            ydata = np.zeros(self.sams[i].adata.shape[0])
        else:
            xdata = self.sams[i].adata.obsm[p][:,0]
            ydata = self.sams[i].adata.obsm[p][:,1]

        if i < len(self.stab.children):
            f1 = self.stab.children[i]
            f1.update_layout(autosize=False)
            f1.data=[]
        else:
            f1 = go.FigureWidget()

        f1.add_scattergl(x=xdata, y=ydata);

        f1.for_each_trace(self.init_graph);
        f1.update_layout(
            margin_l=0,margin_r=0,margin_t=40,margin_b=0,
            width = 600, height=600,
            xaxis_ticks='',
            xaxis_showticklabels=False,
            #xaxis_showgrid=False,
            #xaxis_zeroline=False,
            #yaxis_showgrid=False,
            #yaxis_zeroline=False,
            yaxis_ticks='',
            yaxis_showticklabels=False,
            autosize=True,
            dragmode='select'
        )
        f1.update_yaxes(autorange=True)
        f1.update_xaxes(autorange=True)

        f1.data[0].text = list(self.active_labels[-1])
        f1.data[0].hoverinfo = 'text'
        f1.set_trait('_config',{'displayModeBar':True, 'scrollZoom':True,'displaylogo':False,'edits':{'titleText':False}})

        if i >= len(self.stab.children):
            self.stab.children += (f1,)
            #self.stab.set_trait('selected_index',i)

        if type(self.ds[i]) is int:
            d = Event(source = f1, watched_events = ['keydown'])
            self.ds[i] = d
            d.on_dom_event(self.handle_events)

        self.stab.set_title(i,title)

    def handle_events(self,event):
        if event['type'] == 'keydown':
            key = event['key']
            if key == 'ArrowRight':
                self.cs_box.children[11].children[1].set_trait('value',self.cs_box.children[11].children[1].value+1)
            elif key == 'ArrowLeft':
                x = self.cs_box.children[11].children[1].value-1
                if x < 0:
                    x=0
                self.cs_box.children[11].children[1].set_trait('value',x)

            elif key == 'shift':
                self.irm_genes(None)
            elif key == 'enter':
                self.ism_genes(None)
            elif key == 'x':
                self.unselect_all(None);
            elif key == 'c':
                self.select_all(None);
            elif key == 'v':
                self.reset_view(None);
            elif key == 'a':
                self.cs_box.children[6].children[2].set_trait('value',not self.cs_box.children[6].children[2].value)

    def close_tab(self,event):
        I = self.stab.selected_index
        if I > 0:
            self.stab.set_trait('selected_index',I-1)

            titles = []
            for i in range(len(self.stab.children)):
                titles.append(self.stab.get_title(i))
            del titles[I]

            self.ds[I].close()
            self.stab.children[I].close()
            t= list(self.stab.children)
            del t[I]
            self.stab.children=t
            del self.ds[I]
            del self.sams[I]
            del self.selected[I]
            del self.selected_cells[I]
            del self.active_labels[I]
            del self.marker_genes[I]
            del self.marker_genes_tt[I]

            for i in range(1,len(self.stab.children)):
                self.stab.set_title(i,titles[i])

    """ BEGIN PREPROCESS INIT"""
    def init_preprocess(self):
        pdata = widgets.Button(
            description = 'Process data',
            disabled = False,
            tooltip = 'Process data',
            icon = ''
        )
        pdata.on_click(self.preprocess_sam)
        fgenes = widgets.Checkbox(
            value=self.preprocess_args.get('filter_genes',True),
            description='Filter genes'
        )
        fgenes.observe(self.pp_filtergenes,names = 'value')
        dfts = widgets.Button(
            description = 'Set defaults',
            disabled = False,
            tooltip = 'Set default options',
            icon = ''
        )
        dfts.on_click(self.set_pp_defaults)

        l1 = widgets.Label('Expr threshold:')
        expr_thr = widgets.FloatSlider(
            value=self.preprocess_args.get('thresh',0.01),
            min=0,
            max=0.1,
            step=0.005,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='2f',
            style = {'description_width': 'initial'}
        )
        expr_thr.observe(self.et_update, names='value')

        l2 = widgets.Label('Min expr:')
        min_expr = widgets.FloatSlider(
            value=self.preprocess_args.get('min_expression',1),
            min=0,
            max=6.0,
            step=0.02,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='1f',
            style = {'description_width': 'initial'}
        )
        min_expr.observe(self.me_update, names='value')

        init = self.preprocess_args.get('sum_norm','None')
        if init is None:
            init = 'None'
        sumnorm = widgets.Dropdown(
            options=['cell_median', 'gene_median', 'None'],
            value=init,
            description='Library normalization:',
            disabled=False,
            style = {'description_width': 'initial'}
        )
        sumnorm.observe(self.sumnorm_submit, 'value')

        init = self.preprocess_args.get('norm','log')
        if init is None:
            init = 'None'
        norm = widgets.Dropdown(
            options=['log', 'ftt', 'None'],
            value=init,
            description='Data normalization:',
            disabled=False,
            style = {'description_width': 'initial'}
        )
        norm.observe(self.norm_submit, 'value')
        pp = widgets.VBox([pdata,
                           widgets.HBox([dfts,fgenes]),
                           norm,
                           sumnorm,
                           widgets.HBox([l1,expr_thr]),
                           widgets.HBox([l2,min_expr])])
        #pp = widgets.Box([pdata,dfts,ldata,expr_thr,min_expr,sumnorm,norm])
        return pp
    def me_update(self,val):
        self.preprocess_args['min_expression']=val['new']
    def et_update(self,val):
        self.preprocess_args['thresh']=val['new']
    def sumnorm_submit(self,txt):
        if txt['new'] == 'None':
            self.preprocess_args['sum_norm']=None
        else:
            self.preprocess_args['sum_norm']=txt['new']

    def set_pp_defaults(self,event):
        self.preprocess_args = self.preprocess_args_init.copy()
        fgenes = self.pp_box.children[1].children[1] # checkbox
        norm = self.pp_box.children[2]
        sumnorm = self.pp_box.children[3]
        expr_thr = self.pp_box.children[4].children[1] # expr_threshold
        min_expr = self.pp_box.children[5].children[1] # min_expr

        init = self.preprocess_args.get('min_expression',1)
        min_expr.set_trait('value',init)

        init = self.preprocess_args.get('thresh',0.01)
        expr_thr.set_trait('value',init)

        init = self.preprocess_args.get('sum_norm','None')
        if init is None:
            sumnorm.set_trait('value','None')
        else:
            sumnorm.set_trait('value',init)

        init = self.preprocess_args.get('norm','log')
        if init is None:
            norm.set_trait('value','None')
        else:
            norm.set_trait('value',init)

        init = self.preprocess_args.get('filter_genes',True)
        fgenes.set_trait('value',True)
        self.preprocess_args['filter_genes']=init

    def norm_submit(self,txt):
        if txt['new'] == 'None':
            self.preprocess_args['norm']=None
        else:
            self.preprocess_args['norm']=txt['new']

    def pp_filtergenes(self,event):
        t = self.preprocess_args.get('filter_genes',True)
        self.preprocess_args['filter_genes']=not t

    def preprocess_sam(self,event):
        i = self.stab.selected_index
        self.sams[i].preprocess_data(**self.preprocess_args)
    """ END PREPROCESS INIT"""

    """ BEGIN RUN INIT"""
    def init_run_sam(self):
        runb = widgets.Button(
            description = 'Run SAM',
            disabled = False,
            tooltip = 'Run SAM on the currently selected cells. Enter the title of the new tab in the text box on the right',
            icon = ''
        )
        runb.on_click(self.subcluster)
        title = widgets.Text(
            value = '',
        )
        wpca = widgets.Checkbox(
            value=self.run_args.get('weight_PCs',True),
            description='Weight PCs'
        )
        wpca.observe(self.weightpcs)

        dfts = widgets.Button(
            description = 'Set defaults',
            disabled = False,
            tooltip = 'Set default options',
            icon = ''
        )
        dfts.on_click(self.set_run_defaults)

        l1 = widgets.Button(
            description = '# genes:',
            disabled = True,
            tooltip = 'The number of highest-weighted genes to select each iteration of SAM.',
            icon = ''
        )
        init = self.run_args.get('n_genes',3000)
        if init is None:
            init = 3000
        ngenes = widgets.FloatSlider(
            value=init,
            min=100,
            max=self.sam.adata.shape[1],
            step=100,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='5d',
            style = {'description_width': 'initial'}
        )
        ngenes.observe(self.ngenes_update, names='value')

        l2 = widgets.Button(
            description = '# PCs:',
            disabled = True,
            tooltip = 'The number of principal components to select in PCA.',
            icon = ''
        )
        init = self.run_args.get('npcs',150)
        if init is None:
            init = 150
        npcs = widgets.FloatSlider(
            value=init,
            min=10,
            max=500,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style = {'description_width': 'initial'}
        )
        npcs.observe(self.npcs_update, names='value')

        l3 = widgets.Button(
            description = '# neighbors:',
            disabled = True,
            tooltip = 'The number of nearest neighbors to identify for each cell.',
            icon = ''
        )
        init = self.run_args.get('k',20)
        if init is None:
            init = 20
        knn = widgets.FloatSlider(
            value=init,
            min=5,
            max=200,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style = {'description_width': 'initial'}
        )
        knn.observe(self.knn_update, names='value')

        l4 = widgets.Button(
            description = 'num_norm_avg:',
            disabled = True,
            tooltip = 'The top \'num_norm_avg\' dispersions are averaged to determine the '
            'normalization factor when calculating the weights. This prevents '
            'genes with large spatial dispersions from skewing the distribution '
            'of weights.',
            icon = ''
        )
        init = self.run_args.get('num_norm_avg',50)
        if init is None:
            init = 50
        nna = widgets.FloatSlider(
            value=init,
            min=1,
            max=200,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style = {'description_width': 'initial'}
        )
        nna.observe(self.nna_update, names='value')

        init = self.run_args.get('preprocessing','Normalizer')
        if init is None:
            init = 'None'
        norm = widgets.Dropdown(
            options=['StandardScaler', 'Normalizer', 'None'],
            value=init,
            description='Preprocessing:',
            disabled=False,
            style = {'description_width': 'initial'}
        )
        norm.observe(self.rnorm_update,'value')

        init = self.run_args.get('distance','correlation')
        if init is None:
            init = 'correlation'
        distance = widgets.Dropdown(
            options=['correlation', 'euclidean'],
            value=init,
            description='Distance:',
            disabled=False,
            style = {'description_width': 'initial'}
        )
        distance.observe(self.dist_update,'value')

        init = self.run_args.get('projection','umap')
        if init is None:
            init = 'umap'
        proj = widgets.Dropdown(
            options=['umap', 'tsne', 'diff_umap'],
            value=init,
            description='Projection:',
            disabled=False,
            style = {'description_width': 'initial'}
        )
        proj.observe(self.proj_update,'value')

        rs = widgets.VBox([widgets.HBox([runb,title]),
                           widgets.HBox([dfts,wpca]),
                           widgets.HBox([l1,ngenes]),
                           widgets.HBox([l2,npcs]),
                           widgets.HBox([l3,knn]),
                           widgets.HBox([l4,nna]),
                           norm,
                           distance,
                           proj])
        return rs
    def set_run_defaults(self,event):
        self.run_args = self.run_args_init.copy()
        #run
        #defaults,wpca
        #,ngenes
        #,npcs
        #,knn
        #,nna
        #norm,dist,proj
        wpca = self.rs_box.children[1].children[1]
        ngenes = self.rs_box.children[2].children[1]
        npcs = self.rs_box.children[3].children[1]
        knn = self.rs_box.children[4].children[1]
        nna = self.rs_box.children[5].children[1]
        rnorm = self.rs_box.children[6]
        dist = self.rs_box.children[7]
        proj = self.rs_box.children[8]

        init = self.run_args.get('num_norm_avg',50)
        nna.set_trait('value',init)
        init = self.run_args.get('k',20)
        knn.set_trait('value',init)

        init = self.run_args.get('npcs',150)
        if init is None:
            init = 150;
        npcs.set_trait('value',init)

        init = self.run_args.get('n_genes',3000)
        if init is None:
            init = 3000;
        ngenes.set_trait('value',init)

        init = self.run_args.get('preprocessing','Normalizer')
        if init is None:
            init=''
        rnorm.set_trait('value',init)
        init = self.run_args.get('distance','correlation')
        dist.set_trait('value',init)
        init = self.run_args.get('projection','umap')
        proj.set_trait('value',init)

        init = self.run_args.get('weight_PCs',True)
        wpca.set_trait('value',init)
        self.run_args['weight_PCs'] = init

    def weightpcs(self,event):
        t = self.run_args.get('weight_PCs',True)
        self.run_args['weight_PCs']=not t
    def npcs_update(self,val):
        self.run_args['npcs']=int(val['new'])
    def nna_update(self,val):
        self.run_args['num_norm_avg']=int(val['new'])
    def knn_update(self,val):
        self.run_args['k']=int(val['new'])
    def ngenes_update(self,val):
        self.run_args['n_genes']=int(val['new'])

    def rnorm_update(self,txt):
        if txt['new'] == 'None':
            self.run_args['preprocessing']=None
        else:
            self.run_args['preprocessing']=txt['new']
    def proj_update(self,txt):
        self.run_args['projection']=txt['new']
    def dist_update(self,txt):
        self.run_args['distance']=txt['new']

    def subcluster(self,event):

        execute=False
        i = self.stab.selected_index

        selected = self.selected[i]
        selected_cells = self.selected_cells[i]

        sam = self.sams[i]
        if not np.all(selected) and selected.sum() > 0:
            sam_subcluster = SAM(counts = sam.adata_raw[
                    selected_cells,:].copy())

            sam_subcluster.adata_raw.obs = sam.adata[
                            selected_cells,:].obs.copy()
            sam_subcluster.adata.obs = sam.adata[
                            selected_cells,:].obs.copy()

            sam_subcluster.adata_raw.obsm = sam.adata[
                            selected_cells,:].obsm.copy()
            sam_subcluster.adata.obsm = sam.adata[
                            selected_cells,:].obsm.copy()

            sam_subcluster.preprocess_data(**self.preprocess_args)
            self.out.clear_output()
            with self.out:
                sam_subcluster.run(**self.run_args);
            self.sams.append(sam_subcluster)
            self.selected.append(np.ones(sam_subcluster.adata.shape[0]).astype('bool'))
            self.selected_cells.append(np.array(list(sam_subcluster.adata.obs_names)))
            self.active_labels.append(np.zeros(sam_subcluster.adata.shape[0]))
            self.marker_genes.append(np.array(list(sam_subcluster.adata.var_names))[np.argsort(-sam_subcluster.adata.var['weights'].get_values())])
            self.marker_genes_tt.append('Genes ranked by SAM weights.')
            self.ds.append(0)
            i = len(self.sams)-1
            execute=True

        elif np.all(selected):
            self.out.clear_output()
            with self.out:
                sam.run(**self.run_args)
            self.marker_genes[i] = np.array(list(sam.adata.var_names))[np.argsort(-sam.adata.var['weights'].get_values())]
            self.marker_genes_tt[i] = 'Genes ranked by SAM weights.'
            execute=True

        if execute:
            self.create_plot(i,self.SamPlot.children[1].children[1].children[0].children[1].value)
            self.update_dropdowns(i)

    """ END RUN INIT"""

    """ BEGIN CS INIT"""
    def init_cs(self):
        cpm = widgets.Dropdown(
               options = ['UMAP','t-SNE','Diffusion map','Diffusion UMAP'],
               value = 'UMAP',
               description = '',
               layout={'width':'60%'}
        )

        cp = widgets.Button(
               description = 'Compute projection',
               tooltip = 'Compute a 2D projection using the selected method in the dropdown menu to the left.',
            layout={'width':'40%'}
        )
        cp.on_click(self.compute_projection)

        initl = list(self.sams[0].adata.obsm.keys())
        initl = [""] + initl
        if 'X_umap' in initl:
            init = 'X_umap'
        else:
            init = initl[0]

        dpm = widgets.Dropdown(
               options = initl,
               value = init,
               description = '',
            layout={'width':'60%'}
        )
        dp = widgets.Button(
               description = 'Display projection',
               tooltip = 'Display the 2D projection selected in the dropdown menu to the left.',
            layout={'width':'40%'}
        )
        dp.on_click(self.display_projection)

        clm = widgets.Dropdown(
               options = ['Louvain cluster','Density cluster','Hdbscan cluster','Kmeans cluster','Leiden cluster'],
               value = 'Leiden cluster',
               description = '',
            layout={'width':'60%'}
        )
        clm.observe(self.rewire_cluster_slider,'value')
        cl = widgets.Button(
               description = 'Cluster',
               tooltip = 'Cluster the data using the selected method in the dropdown menu to the left.',
            layout={'width':'40%'}
        )
        cl.on_click(self.cluster_data)

        l = widgets.Label('Leiden \'res\'',
                         layout={'width':'20%'})
        cslider = widgets.FloatSlider(
            value=1,
            min=0.1,
            max=10,
            step=0.1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='2f',
            description = '',
            style = {'description_width': 'initial'},
            layout={'width':'80%'}
        )

        initl = list(self.sams[0].adata.obs.keys())
        initl = [''] + initl
        dam = widgets.Dropdown(
               options = initl,
               value = initl[0],
               description = '',
            layout={'width':'60%'}
        )
        da = widgets.Button(
               description = 'Disp obs ann',
               tooltip = 'Overlay the annotations selected in the dropdown menu to the left.',
            layout={'width':'40%'}
        )
        da.on_click(self.display_annotation)

        dv = widgets.Button(
               description = 'Disp var ann',
               tooltip = 'Include the \'var\' annotation for the corresponding gene in the title of the gene expression plots.',
            layout={'width':'40%'}
        )
        dv.on_click(self.display_var_annotation)

        initl = list(self.sams[0].adata.var.keys())
        initl = [''] + initl
        dvm = widgets.Dropdown(
               options = initl,
               value = initl[0],
               description = '',
            layout={'width':'60%'}
        )

        irm = widgets.Button(
               description = 'Find markers (RF)',
               tooltip = 'Identify marker genes of selected cells using a random forest classifier.',
            layout={'width':'40%'}
        )
        irm.on_click(self.irm_genes)
        ism = widgets.Button(
               description = 'Find markers (SW)',
               tooltip = 'Identify marker genes of selected cells by finding genes with large SAM weights among the selected cells.',
            layout={'width':'40%'}
        )
        ism.on_click(self.ism_genes)
        us = widgets.Button(
               description = 'Unselect all (x)',
               tooltip = 'Unselect all cells. Pressing \'x\' on the keyboard while hovering over the scatter plot will do the same thing.',
            layout={'width':'34%'}
        )
        usa = widgets.Button(
               description = 'Select all (c)',
               tooltip = 'Select all cells. Pressing \'c\' on the keyboard while hovering over the scatter plot will do the same thing.',
            layout={'width':'33%'}
        )
        res = widgets.Button(
               description = 'Reset view (v)',
               tooltip = 'Resets the current plot to the defautl view. Pressing \'v\' on the keyboard while hovering over the scatter plot'
                          ' will do the same thing.',
            layout={'width':'33%'}
        )
        us.on_click(self.unselect_all)
        usa.on_click(self.select_all)
        res.on_click(self.reset_view)

        avg = widgets.Checkbox(
               value=True,
               description = 'Avg expr',
            layout={'width':'50%'}
        )
        lann = widgets.Button(
            description='Annotate',
            tooltip = ('Enter the key of the \'obs\' annotation vector you wish to modify.'
            ' If the key does not exist, a new vector in \'obs\' will be created. Enter the'
            ' label you want to annotate the selected cells with in the right.'),
            disabled=True,
            layout={'width':'20%'}
        )
        anno_name = widgets.Text(
            value = '',
            layout={'width':'40%'}
        )
        anno = widgets.Text(
            value = '',
            layout={'width':'40%'}
        )
        anno.on_submit(self.annotate_pop)

        lgsm = widgets.Button(
            description = 'Get similar genes',
            tooltip = 'Rank genes in order of decreasing spatially averaged correlation with respect to the input gene.',
            disabled=True,
            layout={'width':'30%'}
        )
        gsm = widgets.Text(
            value = '',
            layout={'width':'70%'}
        )
        gsm.on_submit(self.get_similar_genes)
        lshg = widgets.Button(
            description = 'Show expressions',
            tooltip = 'Overlay the input gene expressions over the scatter plot. Only a substring of the gene need be provided.',
            disabled=True,
            layout={'width':'30%'}
        )
        shg = widgets.Text(
            value = '',
            layout={'width':'70%'}
        )
        shg.on_submit(self.show_expression)

        lgenes = widgets.Button(
            description = 'Ranked genes',
            tooltip = 'Genes ranked by SAM weights.',
            disabled=True,
            layout={'width':'30%'})
        rgenes = widgets.IntSlider(
            value=0,
            min=0,
            max=50,
            step=1,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            description = '',
            style = {'description_width': 'initial'},
            layout={'width':'70%'}
        )
        rgenes.observe(self.gene_update,'value')

        lthr = widgets.Button(
            description = 'Threshold expr',
            tooltip = 'Select genes with greater than the chosen expression threshold.',
            disabled=True,
            layout={'width':'30%'})
        thr = widgets.FloatSlider(
            value=0,
            min=0,
            max=0,
            step=0.05,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            description = '',
            style = {'description_width': 'initial'},
            layout={'width':'70%'}
        )
        thr.observe(self.threshold_selection,'value')

        lsf = widgets.Button(
            description = 'Save',
            tooltip = 'Save the current SAM object. Filenames should end with .h5ad or .p.',
            disabled=True,
            layout={'width':'30%'})

        sf = widgets.Text(
            value = '',
            layout={'width':'70%'}
        )
        sf.on_submit(self.save_data)

        close = widgets.Button(
            description = 'Close current tab',
            tooltip = 'Closes the currently open subclustering tab.',
            disabled=False)#layout={'width':'30%'})
        close.on_click(self.close_tab)

        amslider = widgets.Button(
            description = 'Opacity',
            tooltip = 'Changes the opacities of unselected points in the current plot.',
            disabled=True,
            layout={'width':'30%'})
        aslider = widgets.FloatSlider(
            value=0.5,
            min=0.,
            max=1,
            step=0.01,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='2f',
            description = '',
            style = {'description_width': 'initial'},
            layout={'width':'70%'}
        )
        aslider.observe(self.change_alpha,'value')

        lmslider = widgets.Button(
            description = 'Marker size',
            tooltip = 'Changes the marker sizes in the current plot.',
            disabled=True,
            layout={'width':'30%'})
        mslider = widgets.FloatSlider(
            value=5,
            min=0.1,
            max=20,
            step=0.1,
            disabled=False,
            continuous_update=True,
            orientation='horizontal',
            readout=True,
            readout_format='2f',
            description = '',
            style = {'description_width': 'initial'},
            layout={'width':'70%'}
        )
        mslider.observe(self.change_msize,'value')

        return widgets.VBox([
            widgets.HBox([cp,cpm]),
            widgets.HBox([dp,dpm]),
            widgets.HBox([cl,clm]),
            widgets.HBox([l,cslider]),
            widgets.HBox([da,dam]),
            widgets.HBox([dv,dvm]),
            widgets.HBox([irm,ism,avg]),
            widgets.HBox([us,usa,res]),
            widgets.HBox([lann,anno_name,anno]),
            widgets.HBox([lgsm,gsm]),
            widgets.HBox([lshg,shg]),
            widgets.HBox([lgenes,rgenes]),
            widgets.HBox([lthr,thr]),
            widgets.HBox([lsf,sf]),
            widgets.HBox([amslider,aslider]),
            widgets.HBox([lmslider,mslider]),
            close
        ])
    def reset_view(self,event):
        self.create_plot(self.stab.selected_index,self.stab.get_title(self.stab.selected_index))

    def save_data(self,path):
        path=path.value
        if path != '':
            if path.split('.')[-1] == 'h5ad':
                s = self.sams[self.stab.selected_index]
                s.save_anndata(path,data='adata')
            elif path.split('.')[-1] == 'p':
                s = self.sams[self.stab.selected_index]
                s.save(path)
    def show_expression(self,text):
        if type(text) is not str:
            gene = text.value
        else:
            gene = text

        s = self.sams[self.stab.selected_index]
        if gene != '':

            try:
                genes = ut.search_string(np.array(list(s.adata.var_names)),gene, case_sensitive=True)[0]

                if genes is not -1:
                    gene=genes[0]

                if self.cs_box.children[6].children[2].value:
                    x = s.adata[:,gene].layers['X_knn_avg']
                    if sp.issparse(x):
                        a = x.A.flatten()
                    else:
                        a = x.flatten()
                else:
                    x = s.adata_raw[:,gene].X
                    if sp.issparse(x):
                        a = x.A.flatten()
                    else:
                        a = x.flatten()

                if(self.cs_box.children[6].children[2].value):
                    if a.sum() == 0:
                        x = s.adata_raw[:,gene].X
                        if sp.issparse(x):
                            a = x.A.flatten()
                        else:
                            a = x.flatten()

                        norm = self.preprocess_args.get('norm','log')
                        norm = 'log'
                        if(norm.lower() == 'log'):
                            a = np.log2(a + 1)
                        elif(norm.lower() == 'ftt'):
                            a = np.sqrt(a) + np.sqrt(a+1) - 1
                        elif(norm.lower() == 'asin'):
                            a = np.arcsinh(a)
                else:
                    norm = self.preprocess_args.get('norm','log')
                    norm = 'log'
                    if(norm.lower() == 'log'):
                        a = np.log2(a + 1)
                    elif(norm.lower() == 'ftt'):
                        a = np.sqrt(a) + np.sqrt(a+1) - 1
                    elif(norm.lower() == 'asin'):
                        a = np.arcsinh(a)

                if self.GENE_KEY!='':
                    title = gene+'; ' +str(s.adata.var[self.GENE_KEY].T[gene])
                else:
                    title=gene

                self.select_all(None)
                self.update_colors_expr(a,title)
                self.gene_expression = a
            except IndexError:
                0; # do nothing

    def get_similar_genes(self,txt):
        gene = txt.value

        s = self.sams[self.stab.selected_index]

        genes = ut.search_string(np.array(list(s.adata.var_names)),gene, case_sensitive=True)[0]
        if genes is not -1:
            gene = genes[0]
        else:
            return; #quit

        markers = ut.find_corr_genes(s,gene).flatten()
        _,i = np.unique(markers,return_index=True)
        markers=markers[np.sort(i)]
        self.marker_genes[self.stab.selected_index] = markers

        self.cs_box.children[11].children[1].set_trait('value',0)
        self.marker_genes_tt[self.stab.selected_index]='Ranked genes from most to least spatially correlated with '+gene+'.'
        self.cs_box.children[11].children[0].set_trait('tooltip',self.marker_genes_tt[self.stab.selected_index])
        self.show_expression(str(gene))

    def annotate_pop(self,text):
        text = text.value
        text_name = self.cs_box.children[8].children[1].value
        selected = self.selected[self.stab.selected_index]
        selected_cells = self.selected_cells[self.stab.selected_index]

        for it,s in enumerate(self.sams):
            x1 = np.array(list(s.adata.obs_names))

            if text!='' and text_name != '' and selected.sum()!=selected.size:
                if text_name in list(s.adata.obs.keys()):
                    a = s.adata.obs[text_name].get_values().copy().astype('<U100')
                    a[np.in1d(x1,selected_cells)] = text
                    s.adata.obs[text_name] = pd.Categorical(a)

                else:
                    a = np.zeros(s.adata.shape[0],dtype='<U100')
                    a[:]=""
                    a[np.in1d(x1,selected_cells)] = text
                    s.adata.obs[text_name] = pd.Categorical(a)

            self.update_dropdowns(it)

    def unselect_all(self,event):
        self.selected[self.stab.selected_index][:] = False
        self.selected_cells[self.stab.selected_index] = np.array([])
        self.stab.children[self.stab.selected_index].data[0].selectedpoints=np.array([])
    def select_all(self,event):
        self.selected[self.stab.selected_index][:] = True
        self.selected_cells[self.stab.selected_index] = np.array(list(self.sams[self.stab.selected_index].adata.obs_names))
        self.stab.children[self.stab.selected_index].data[0].selectedpoints=np.arange(self.selected[self.stab.selected_index].size)
        self.stab.children[self.stab.selected_index].data[0].marker.opacity=1

    def ism_genes(self,event):
        selected = self.selected[self.stab.selected_index]
        s = self.sams[self.stab.selected_index]
        if not np.all(selected) and selected.sum() > 0:
            l = s.adata.layers['X_knn_avg']
            m = l.mean(0).A.flatten()
            ms = l[selected,:].mean(0).A.flatten()
            lsub = l[selected,:]
            lsub.data[:] = lsub.data**2
            ms2 = lsub.mean(0).A.flatten()
            v = ms2 - 2*ms*m + m**2
            wmu = np.zeros(v.size)
            wmu[m>0] = v[m>0] / m[m>0]

            self.marker_genes[self.stab.selected_index] = np.array(list(s.adata.var_names[np.argsort(-wmu)]))
            if self.cs_box.children[11].children[1].value != 0:
                self.cs_box.children[11].children[1].set_trait('value',0)
            else:
                self.show_expression(str(self.marker_genes[self.stab.selected_index][0]))
            self.marker_genes_tt[self.stab.selected_index] = 'Ranked genes according to spatial SAM weights among selected cells.'
            self.cs_box.children[11].children[0].set_trait('tooltip',self.marker_genes_tt[self.stab.selected_index])
    def irm_genes(self,event):
        selected = self.selected[self.stab.selected_index]
        s = self.sams[self.stab.selected_index]
        if not np.all(selected) and selected.sum() > 0:
            a = np.zeros(s.adata.shape[0])
            a[selected]=1
            markers,_ = s.identify_marker_genes_rf(labels = a,clusters = 1)
            self.marker_genes[self.stab.selected_index]=markers[1]
            if self.cs_box.children[11].children[1].value != 0:
                self.cs_box.children[11].children[1].set_trait('value',0)
            else:
                self.show_expression(str(self.marker_genes[self.stab.selected_index][0]))
            self.marker_genes_tt[self.stab.selected_index]='Ranked genes according to random forest classification.'
            self.cs_box.children[11].children[0].set_trait('tooltip',self.marker_genes_tt[self.stab.selected_index])

    def gene_update(self,val):
        val=val['new']
        markers = self.marker_genes[self.stab.selected_index]

        if int(val) < markers.size:
            gene = markers[int(val)]
        else:
            gene = markers[-1]
        self.cs_box.children[10].children[1].set_trait('value',gene)
        self.show_expression(str(gene))

    def update_dropdowns(self,i):
        s = self.sams[i]
        self.cs_box.children[1].children[1].options = list(s.adata.obsm.keys())
        self.cs_box.children[4].children[1].options = [''] + list(s.adata.obs.keys())
        self.cs_box.children[5].children[1].options = [''] + list(s.adata.var.keys())
    def on_switch_tabs(self,event):
        self.update_dropdowns(self.stab.selected_index)
        self.cs_box.children[11].children[0].set_trait('tooltip',self.marker_genes_tt[self.stab.selected_index])


    def display_var_annotation(self, event):
        self.GENE_KEY=self.cs_box.children[5].children[1].value

    def display_annotation(self, event):
        key = self.cs_box.children[4].children[1].value
        labels = self.sams[self.stab.selected_index].adata.obs[key].get_values()
        self.active_labels[self.stab.selected_index] = labels
        self.selected[self.stab.selected_index][:]=True
        self.stab.children[self.stab.selected_index].data[0].selectedpoints = np.where(
                        self.selected[self.stab.selected_index][:])[0]
        self.update_colors_anno(labels)

    def update_colors_expr(self,a,title):
        f1 = self.stab.children[self.stab.selected_index]
        f1.update_traces(marker = dict(color = a,colorscale='spectral',reversescale=True,
                                       showscale=True,colorbar_ticks='outside',
                                       colorbar_tickmode='auto',colorbar_title='',
                                       opacity=1))

        f1.update_layout(hovermode = 'closest', title = title)
        self.stab.children[self.stab.selected_index].data[0].text = list(a)
        self.stab.children[self.stab.selected_index].data[0].hoverinfo = 'text'


        slider = self.cs_box.children[12].children[1]
        slider.set_trait('max',a.max()+(a.max()-a.min())/100)
        slider.set_trait('min',0)
        slider.set_trait('step',(a.max()-a.min())/100)
        slider.set_trait('value',0)

    def update_colors_anno(self,labels):
        self.labels=labels
        nlabels = np.unique(labels).size
        if nlabels == 1:
            x = 'spectral'
        elif nlabels <= 2:
            x = cl.scales['3']['qual']['Paired'][:nlabels]
        elif nlabels <= 11:
            nlabels = str(nlabels)
            x = cl.scales[nlabels]['qual']['Paired']
        else:
            nlabels = '11'
            x = cl.scales[nlabels]['qual']['Paired']
            x = cl.interp(x,int(nlabels))

        lbls,inv = np.unique(labels,return_inverse=True)

        if type(labels.flatten()[0]) is str:
            tickvals=np.arange(lbls.size)
            ticktext=list(lbls)
        else:
            tickvals=[0,lbls.size-1]
            ticktext=[lbls.min(),lbls.max()]

        title = self.cs_box.children[4].children[1].value.split('_clusters')[0]

        f1 = self.stab.children[self.stab.selected_index]
        f1.update_traces(marker = dict(color = inv,colorscale=x,
                                       showscale=True,colorbar_ticks='outside',
                                       colorbar_tickmode='array',colorbar_title='',
                                       colorbar_tickvals=tickvals,
                                       colorbar_ticktext=ticktext,
                                       opacity=1))
        self.stab.children[self.stab.selected_index].layout.title=title
        self.stab.children[self.stab.selected_index].data[0].text = list(lbls[inv])
        self.stab.children[self.stab.selected_index].data[0].hoverinfo = 'text'
        self.stab.children[self.stab.selected_index].layout.hovermode = 'closest'


        slider = self.cs_box.children[12].children[1]
        slider.set_trait('min',0)
        slider.set_trait('max',0)
        slider.set_trait('value',0)

    def threshold_selection(self, val):
        val = val['new']
        s = self.sams[self.stab.selected_index]

        self.selected[self.stab.selected_index][:]=False
        self.selected[self.stab.selected_index][self.gene_expression>=val]=True
        self.selected_cells[self.stab.selected_index]=np.array(list(s.adata.obs_names))[self.selected[self.stab.selected_index]]
        self.stab.children[self.stab.selected_index].data[0].selectedpoints = np.where(self.selected[self.stab.selected_index])[0]

    def rewire_cluster_slider(self,event):
        x = self.cs_box.children[3].children[1]
        l = self.cs_box.children[3].children[0]
        val = event['new']
        if val == 'Kmeans cluster':
            x.set_trait('min',2)
            x.set_trait('max',50)
            x.set_trait('value',6)
            x.set_trait('step',1)
            l.set_trait('value','Kmeans \'k\'')

        elif val == 'Louvain cluster':
            x.set_trait('min',0.1)
            x.set_trait('max',10)
            x.set_trait('value',1)
            x.set_trait('step',.1)
            l.set_trait('value','Louvain \'res\'')

        elif val == 'Density cluster':
            x.set_trait('min',0.1)
            x.set_trait('max',2)
            x.set_trait('value',0.5)
            x.set_trait('step',0.01)
            l.set_trait('value','Density \'eps\'')

        elif val == 'Hdbscan cluster':
            l.set_trait('value','Hdbscan \'N/A\'')

        elif val == 'Leiden cluster':
            x.set_trait('min',.1)
            x.set_trait('max',10)
            x.set_trait('value',1)
            x.set_trait('step',.1)
            l.set_trait('value','Leiden \'res\'')

    def cluster_data(self,event):
        s = self.sams[self.stab.selected_index]
        val = self.cs_box.children[2].children[1].value
        eps = self.cs_box.children[3].children[1].value;
        if val == 'Kmeans cluster':
            s.kmeans_clustering(int(eps))

        elif val == 'Louvain cluster':
            s.louvain_clustering(res = eps)

        elif val == 'Density cluster':
            s.density_clustering(eps = eps)

        elif val == 'Hdbscan cluster':
            s.hdbknn_clustering(k = int(eps))

        elif val == 'Leiden cluster':
            s.leiden_clustering(res = eps)

        self.cs_box.children[4].children[1].options = [''] + list(s.adata.obs.keys())

    def display_projection(self,event):
        s=self.sams[self.stab.selected_index]
        X = s.adata.obsm[self.cs_box.children[1].children[1].value][:,:2]
        self.stab.children[self.stab.selected_index].data[0]['x'] = X[:,0]
        self.stab.children[self.stab.selected_index].data[0]['y'] = X[:,1]

    def compute_projection(self,event):
        i = self.stab.selected_index
        s = self.sams[i]
        val = self.cs_box.children[0].children[1].value
        if val == 'UMAP':
            s.run_umap()
        elif val == 't-SNE':
            s.run_tsne()
        elif val == 'Diffusion map':
            s.run_diff_map()
        elif val == 'Diffusion UMAP':
            s.run_diff_umap()
        self.cs_box.children[1].children[0].options = list(s.adata.obsm.keys())
    """ END CS INIT"""

    def select(self,trace,points,selector):
        if np.array(points.point_inds).size>0:
            self.selected[self.stab.selected_index][np.array(points.point_inds)]=True
            self.selected_cells[self.stab.selected_index] = np.array(list(self.sams[
                                                        self.stab.selected_index].adata.obs_names[
                                                        self.selected[self.stab.selected_index]]))
        trace.selectedpoints = list(np.where(self.selected[self.stab.selected_index])[0])
    def pick_cells(self,trace, points, selector):
        tf = self.selected[self.stab.selected_index][points.point_inds[0]]
        als = self.active_labels[self.stab.selected_index]
        al = als[points.point_inds[0]]

        if tf:
            self.selected[self.stab.selected_index][als==al]=False
        else:
            self.selected[self.stab.selected_index][als==al]=True

        self.selected_cells[self.stab.selected_index] = np.array(list(self.sams[
                                                self.stab.selected_index].adata.obs_names[
                                                self.selected[self.stab.selected_index]]))
        trace.selectedpoints = list(np.where(self.selected[self.stab.selected_index])[0])

    def change_alpha(self,val):
        #m = self.stab.children[self.stab.selected_index].data[0].marker.copy()
        #m['opacity'] = val['new']
        self.stab.children[self.stab.selected_index].data[0].unselected = {'marker':{'opacity':val['new']}}


    def change_msize(self,val):
        val = val['new']
        markers = self.stab.children[self.stab.selected_index].data[0].marker
        markers.size=val

    def init_graph(self,trace):
        trace.mode = 'markers'
        trace.marker.size = 5
        trace.on_selection(self.select)
        trace.on_click(self.pick_cells)
