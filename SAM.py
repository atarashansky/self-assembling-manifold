from anndata import AnnData
import scipy.sparse as sp
import time
from sklearn.preprocessing import Normalizer, StandardScaler
import pickle
import pandas as pd
import utilities as ut
import sklearn.manifold as man
import sklearn.utils.sparsefuncs as sf
import warnings

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import numpy as np

try:
    import matplotlib.pyplot as plt
    PLOTTING = True
except ImportError:
    print('matplotlib not installed. Plotting functions disabled')
    PLOTTING = False


__version__ = '0.4.2'

"""
Copyright 2018, Alexander J. Tarashansky, All rights reserved.
Email: <tarashan@stanford.edu>
"""

class SAM(object):
    """Self-Assembling Manifolds single-cell RNA sequencing analysis tool.

    SAM iteratively rescales the input gene expression matrix to emphasize
    that are spatially variable along the intrinsic manifold of the data.
    It outputs the gene weights, nearest neighbor matrix, distance matrix, and
    a 2D projection.

    Parameters
    ----------
    counts : tuple or list (scipy.sparse matrix, numpy.ndarray,numpy.ndarray), OR
        tuple or list (numpy.ndarray, numpy.ndarray,numpy.ndarray), OR
        pandas.DataFrame, OR
        anndata.AnnData

        If a tuple or list, it should contain the gene expression data (scipy.sparse or 
        numpy.ndarray) matrix (cells x genes), numpy array of gene IDs, and numpy array
        of cell IDs in that order.

        If a pandas.DataFrame, it should be (cells x genes)

        Only use this argument if you want to pass in preloaded data. Otherwise
        use one of the load functions.

    annotations : numpy.ndarray, optional, default None
        A Numpy array of cell annotations.


    Attributes
    ----------
    raw_data: scipy.sparse matrix
        Sparse data structure of raw gene expression counts.
    
    all_cell_names: numpy.array
        Vector of cell IDs corresponding to rows of raw_data
    
    all_gene_names: numpy.array
        Vector of gene IDs corresponding to columns of raw_data

    k: int
        The number of nearest neighbors to identify for each cell
        when constructing the nearest neighbor graph.

    distance: str
        The distance metric used when constructing the cell-to-cell
        distance matrix.
        
    output_vars: dict
        Dictionary of SAM outputs.

    D: scipy.sparse matrix
        The filtered and log-transformed data used for dimensionality reduction.
        Using the gene regression function manipulates this matrix.
    
    D2: scipy.sparse matrix
        The filtered and log-transformed data used for weight calculation.        
    
    gene_names: numpy.ndarray
        A vector of the gene names corresponding to columns of D.

    cell_names: numpy.ndarray
        A vector of the cell names corresponding to rows of D.

    annotations: numpy.ndarray
        A vector of cell annotations if they were provided.

    integer_annotations: numpy.ndarray
        A vector of cell annotations converted to integers.

    D_avg: scipy.sparse matrix
        The k-nearest-neighbor-averaged expression data.

    wPCA_data: numpy.ndarray
        The weighted PCA matrix.

    indices: numpy.ndarray
        Indices sorting the genes according to the weights in descending order
        (i.e. indices[0] is the index of the gene with the largest weight).

    nnm_adj: scipy.sparse matrix
        The nearest neighbor adjacency matrix.

    weights: numpy.ndarray
        A vector of weights for each gene.

    ranked_genes: numpy.ndarray
        The ranked list of genes, i.e. sam.gene_names[sam.indices].

    tsne2d: numpy.ndarray
        The t-SNE embedding.

    umap2d: numpy.ndarray
        The UMAP embedding.

    geneID_groups: list of numpy.ndarray
        Each element of the list contains a vector of gene IDs that are
        correlated with each other along the SAM manifold.
        
    adata: AnnData
        An AnnData object containing the processed data and SAM outputs. Used
        for integration with Scanpy.
        
    """

    def __init__(self, counts=None, annotations=None):
        
        if isinstance(counts, tuple) or isinstance(counts,list):
            self.raw_data,self.all_gene_names,self.all_cell_names = counts
            if isinstance(self.raw_data,np.ndarray):
                self.raw_data = sp.csr_matrix(self.raw_data)
            self.D = self.raw_data.copy()
            self.D2=self.D.copy()
            self.all_gene_names=np.array(list(self.all_gene_names))
            self.all_cell_names=np.array(list(self.all_cell_names))
            self.gene_names=self.all_gene_names.copy()
            self.cell_names=self.all_cell_names.copy()
            self.adata = AnnData(X = self.D2,obs={'obs_names':self.all_cell_names},
                        var = {'var_names':self.gene_names})
        elif isinstance(counts,pd.DataFrame):
            self.raw_data = sp.csr_matrix(counts.values)
            self.all_gene_names = np.array(list(counts.columns.values))
            self.all_cell_names = np.array(list(counts.index.values))
            self.D = self.raw_data.copy()
            self.D2=self.D.copy()
            self.gene_names=self.all_gene_names.copy()
            self.cell_names=self.all_cell_names.copy()
            self.adata = AnnData(X = self.D2,obs={'obs_names':self.all_cell_names},
                        var = {'var_names':self.gene_names})
        elif isinstance(counts,AnnData):
            self.adata = counts
            self.all_cell_names = np.array(list(self.adata.obs_names))
            self.all_gene_names = np.array(list(self.adata.var_names))

            self.raw_data = sp.csr_matrix(self.adata.X)            
            self.D = self.raw_data.copy()
            self.D2 = self.D.copy()
            
            self.gene_names=self.all_gene_names.copy()
            self.cell_names = self.all_cell_names.copy()
            
        elif counts is not None:
            raise Exception("\'counts\' must be either a tuple/list of (data,gene IDs,cell IDs)"
                             "or a Pandas DataFrame of cells x genes")

        if(annotations is not None):
            self.annotations = np.array(list(annotations))
            self.integer_annotations = ut.convert_annotations(self.annotations)
            
            if counts is not None:
                self.adata.obs['ann']=self.annotations
                self.adata.obs['annInt']=self.integer_annotations            
                
        self.analysis_performed = False

        self.output_vars = {}
        

               
    
    def preprocess_data(self, div=1, downsample=0, sum_norm = None,
                        include_genes=None, exclude_genes=None,
                        include_cells=None, exclude_cells=None,
                        norm='log', min_expression=1, thresh=0.01,
                        filter_genes=True):
        
        """Log-normalizes and filters the expression data.

        Parameters
        ----------

        div : float, optional, default 1
            The factor by which the gene expression will be divided prior to
            log normalization.

        downsample : float, optional, default 0
            The factor by which to randomly downsample the data. If 0, the
            data will not be downsampled.

        sum_norm : float, optional, default None
            If specified, the total number of transcripts in each cell will be
            normalized to this value prior to log normalization and filtering.
            Otherwise, nothing happens.

        norm : str, optional, default 'log'
            If 'log', log-normalizes the expression data. If the loaded data is
            already log-normalized, set norm = None.
            
        include_genes : array-like of string, optional, default None
            A vector of gene names or indices that specifies the genes to keep.
            All other genes will be filtered out. Gene names are case-sensitive.
    
        exclude_genes : array-like of string, optional, default None
            A vector of gene names or indices that specifies the genes to exclude.
            These genes will be filtered out. Gene names are case-sensitive.
    
        include_cells : array-like of string, optional, default None
            A vector of cell names that specifies the cells to keep.
            All other cells will be filtered out. Cell names are
            case-sensitive.
    
        exclude_cells : array-like of string, optional, default None
            A vector of cell names that specifies the cells to exclude.
            Thses cells will be filtered out. Cell names are
            case-sensitive.
        
        min_expression : float, optional, default 1
            The threshold above which a gene is considered
            expressed. Gene expression values less than 'min_expression' are
            set to zero.

        thresh : float, optional, default 0.2
            Keep genes expressed in greater than 'thresh'*100 % of cells and
            less than (1-'thresh')*100 % of cells, where a gene is considered
            expressed if its expression value exceeds 'min_expression'.

        filter_genes : bool, optional, default True
            Setting this to False turns off filtering operations aside from
            removing genes with zero expression across all cells. Genes passed
            in exclude_genes or not passed in include_genes will still be
            filtered.

        """
        if(self.raw_data is None):
            print('No data loaded')
            return

        self.D = self.raw_data.copy()    
        if (sum_norm is not None):
            self.D = self.D.multiply(1/self.D.sum(1).A.flatten()[:,None]*sum_norm).tocsr()
        
        if(norm == 'log'):
            self.D.data = np.log2(self.D.data/div+1)
        else:
            self.D.data = self.D.data/div
            
        self.gene_names = self.all_gene_names
        self.cell_names = self.all_cell_names
        
        
        
        if(include_genes is not None):
            include_genes = np.array(include_genes)            
            idx = np.where(np.in1d(self.gene_names, include_genes))[0]
            self.D = self.D[:, idx]
            self.gene_names = self.gene_names[idx]
            
        if(include_cells is not None):
            include_cells = np.array(include_cells)            
            idx2 = np.where(np.in1d(self.cell_names, include_cells))[0]            
            self.D = self.D[idx2, :]
            self.cell_names = self.cell_names[idx2]

        if(exclude_genes is not None):
            exclude_genes = np.array(exclude_genes)            
            idx3 = np.where(np.in1d(self.gene_names, exclude_genes,
                                                            invert=True))[0]
            self.D = self.D[:, idx3]
            self.gene_names = self.gene_names[idx3]
        
        if(exclude_cells is not None):
            exclude_cells = np.array(exclude_cells)            
            idx4 = np.where(np.in1d(self.cell_names, exclude_cells,
                                                            invert=True))[0]            
            self.D = self.D[idx4, :]
            self.cell_names = self.cell_names[idx4]        

        if downsample > 0:
            numcells = int(self.D.shape[0]/downsample)
            rand_ind= np.random.choice(np.arange(self.D.shape[0]),
                                 size=numcells, replace=False)
            
            self.D = self.D[rand_ind, :]
            self.cell_names = self.cell_names[rand_ind]

        else:
            numcells = self.D.shape[0]


        idx5 = np.where(self.D.data <= min_expression)[0]
        self.D.data[idx5]=0
        self.D.eliminate_zeros()
        
        
        if(filter_genes):
            a,ct = np.unique(self.D.nonzero()[1],return_counts=True)
            c = np.zeros(self.D.shape[1])
            c[a]=ct
            
            keep = np.where(np.logical_and(c/self.D.shape[0] > thresh,
                                           c/self.D.shape[0] < 1-thresh))[0]
        
        else:
            keep = np.arange(self.D.shape[1])
            

        self.D = self.D.tocsc()
        self.D = self.D[:,keep]
        self.D2 = self.D
        
        self.gene_names = self.gene_names[keep]
        
        self.adata = AnnData(X = self.D2,obs={'obs_names':self.cell_names},
                        var = {'var_names':self.gene_names})       
        
    def load_data(self, filename, transpose=True,
                            save_sparse_file=True,sep=','):
        """Loads the specified data file. The file can be a table of
        read counts (i.e. '.csv' or '.txt'), with genes as rows and cells
        as columns by default. The file can also be a pickle file (output from
        'save_sparse_data').

        This function that loads the file specified by 'filename'.

        Parameters
        ----------
        filename - string
            The path to the tabular raw expression counts file.

        sep - string, optional, default ','
            The delimeter used to read the input data table. By default
            assumes the input table is delimited by commas.
        
        save_sparse_file - bool, optional, default True
            If True, pickles the sparse data structure, cell names, and gene names
            in the same folder as the original data for faster loading in the
            future. 
            
        transpose - bool, optional, default True
            By default, assumes file is (genes x cells). Set this to False if
            the file has dimensions (cells x genes).
        
        
        """
        if filename.split('.')[-1]=='p':
            self.raw_data,self.all_cell_names,self.all_gene_names = (
                                    pickle.load(open(filename,'rb')))
            
            if(transpose):
                self.raw_data = self.raw_data.T        
        
            #self.filter_data(**kwargs)      
            
        else:    
            df = pd.read_csv(filename, sep=sep, index_col=0)
            if(transpose):
                dataset = df.T
            else:
                dataset = df
            
            self.raw_data = sp.csr_matrix(dataset.values)
            self.all_cell_names = np.array(list(dataset.index.values))
            self.all_gene_names = np.array(list(dataset.columns.values))
            
            #self.filter_data(**kwargs)        
            
            if(save_sparse_file):
                new_sparse_file = filename.split('/')[-1].split('.')[0]        
                path = filename[:filename.find(filename.split('/')[-1])]                        
                self.save_sparse_data(path+new_sparse_file+'_sparse.p')

        self.D = self.raw_data.copy()
        self.D2=self.D.copy()
        self.all_gene_names=np.array(list(self.all_gene_names))
        self.all_cell_names=np.array(list(self.all_cell_names))
        self.gene_names=self.all_gene_names.copy()
        self.cell_names=self.all_cell_names.copy()
        self.adata = AnnData(X = self.D2,obs={'obs_names':self.all_cell_names},
                    var = {'var_names':self.gene_names})

            
    def save_sparse_data(self,fname):
        """Saves the tuple (raw_data,all_cell_names,all_gene_names) in a
        Pickle file.
        
        Parameters
        ----------
        fname - string
            The filename of the output file.        

        """                
        pickle.dump((self.raw_data.T,self.all_cell_names,self.all_gene_names),open(fname,'wb'))

    def load_annotations(self, aname,sep=','):
        """Loads cell annotations.

        Loads the cell annoations specified by the 'aname' path.

        Parameters
        ----------
        aname - string
            The path to the annotations file. First column should be cell IDs
            and second column should be the desired annotations.
            
        """
        ann = pd.read_csv(aname)
        if(ann.shape[1] > 1):
            ann = pd.read_csv(aname, index_col=0)
            if(ann.shape[0] != self.all_cell_names.size):
                ann = pd.read_csv(aname, index_col=0,header=None)
        else:
            if(ann.shape[0] != self.all_cell_names.size):
                ann = pd.read_csv(aname,header=None)            
        try:
            ann = np.array(list(ann.T[self.cell_names].T.values.flatten()))
        except:
            ann=np.array(list(ann.values.flatten()))

        self.annotations = ann
        self.integer_annotations = ut.convert_annotations(self.annotations)
        
        self.adata.obs['ann']=self.annotations
        self.adata.obs['annInt']=self.integer_annotations 
    
    def dispersion_ranking_NN(self, nnm, num_norm_avg=50):
        """Computes the spatial dispersion factors for each gene.      

        Parameters
        ----------
        nnm - scipy.sparse, float
            Square cell-to-cell nearest-neighbor matrix.

        num_norm_avg - int, optional, default 50
            The top 'num_norm_avg' dispersions are averaged to determine the
            normalization factor when calculating the weights. This ensures
            that outlier genes do not significantly skew the weight
            distribution.

        Returns:
        -------
        indices - ndarray, int
            The indices corresponding to the gene weights sorted in decreasing
            order.

        weights - ndarray, float
            The vector of gene weights.
        """

        D_avg = (nnm/self.k).dot(self.D2)
        
        self.D_avg = D_avg
        mu,var = sf.mean_variance_axis(D_avg,axis=0)

        dispersions = np.zeros(var.size)
        dispersions[mu>0] = var[mu>0]/mu[mu>0]
        
        self.dispersions = dispersions.copy()
        
        ma = np.sort(dispersions)[-num_norm_avg:].mean()
        dispersions[dispersions >= ma] = ma

        weights = ((dispersions/dispersions.max())**0.5).flatten()       
        indices = np.argsort(-weights)

        return indices, weights    
    
    def calculate_regression_PCs(self,genes=None,npcs = None, plot=False):
        """Computes the contribution of the gene IDs in 'genes' to each
        principal component (PC) of the filtered expression data as the mean of
        the absolute value of the corresponding gene loadings. High values
        correspond to PCs that are highly correlated with the features in
        'genes'. These PCs can then be regressed out of the data using
        'regress_genes'.
        

        Parameters
        ----------
        genes - numpy.array or list
            Genes for which contribution to each PC will be calculated.

        npcs - int, optional, default None
            How many PCs to calculate when computing PCA of the filtered and
            log-transformed expression data. If None, calculate all PCs.
        
        plot - bool, optional, default False
            If True, plot the scores reflecting how correlated each PC is with
            genes of interest. Otherwise, do not plot anything.
            
        Returns:
        -------
        x - numpy.array
            Scores reflecting how correlated each PC is with the genes of
            interest (ordered by decreasing eigenvalues).
                    
        """
        from sklearn.decomposition import PCA
        
        if npcs is None:
            npcs = self.D.shape[0]
            
        pca= PCA(n_components=npcs)
        pc= pca.fit_transform(self.D.toarray()-self.D.mean(0).A.flatten())
        
        self.regression_pca = pca
        self.regression_pcs = pc
        
        if(genes is not None):
            idx = np.where(np.in1d(self.gene_names,genes))[0]
            sx = pca.components_[:,idx]
            x=np.abs(sx).mean(1)
        
            if plot:
                plt.figure(); plt.plot(x);
            
            return x
        else:
            return

    def regress_genes(self,PCs):
        """Regress out the principal components in 'PCs' from the filtered
        expression data ('SAM.D'). Assumes 'calculate_regression_PCs' has
        been previously called.
        
        Parameters
        ----------
        PCs - int, numpy.array, list
            The principal components to regress out of the expression data.
                   
        """
        
        ind = [PCs]
        ind = np.array(ind).flatten()
        try:
            y = self.D.toarray() - self.regression_pcs[:,ind].dot(
                    self.regression_pca.components_[ind,:]*self.weights_f)
        except:
            y = self.D.toarray() - self.regression_pcs[:,ind].dot(
                    self.regression_pca.components_[ind,:])
            
        self.D = sp.csr_matrix(y)    
        
        self.adata.layers['X_regressed'] = self.D
        
    def kmeans_clustering(self,numc,X=None,npcs = 15):
        """Performs k-means clustering.
        
        Parameters
        ----------
        numc - int
            Number of clusters
        
        npcs - int, optional, default 15
            Number of principal components to use as inpute for k-means clustering.
               
        """
        
        from sklearn.cluster import KMeans
        if X is None:
            X = (self.D_sub-self.D_sub.mean(0)
                                        ).dot(self.pca.components_[:npcs,:].T)
            save=True
        else:
            save=False
        
        cl = KMeans(n_clusters=numc).fit_predict(Normalizer().fit_transform(X))        
        
        if save:
            self.output_vars['kmeans_clusters'] = cl        
            self.adata.obs['kmeans_clusters'] = cl
        else:
            return cl
    
    
    def run(self,
            max_iter=10,
            verbose=True,
            projection='umap',
            stopping_condition=5e-3,
            num_norm_avg=50,
            k = 20,
            distance='correlation',
            preprocessing='Normalizer'):
        
        """Runs the Self-Assembling Manifold algorithm.

        Parameters
        ----------
        k - int, optional, default 20
            The number of nearest neighbors to identify for each cell.
        
        distance : string, optional, default 'correlation'
            The distance metric to use when constructing cell distance
            matrices. Can be any of the distance metrics supported by
            sklearn's 'pdist'.

        max_iter - int, optional, default 10
            The maximum number of iterations SAM will run.

        stopping_condition - float, optional, default 5e-3
            The stopping condition threshold for the RMSE between gene weights in 
            adjacent iterations.

        verbose - bool, optional, default True
            If True, the iteration number and error between gene weights in adjacent
            iterations will be displayed.
            
        projection - str, optional, default 'umap'
            If 'tsne', generates a t-SNE embedding. If 'umap', generates a UMAP
            embedding. Otherwise, no embedding will be generated.
        
        preprocessing - str, optional, default 'Normalizer'
            If 'Normalizer', use sklearn.preprocessing.Normalizer, which
            normalizes expression data prior to PCA such that each cell has
            unit L2 norm. If 'StandardScaler', use
            sklearn.preprocessing.StandardScaler, which normalizes expression
            data prior to PCA such that each gene has zero mean and unit
            variance. Otherwise, do not normalize the expression data. We
            recommend using 'StandardScaler' for large datasets and 'Normalizer'
            otherwise.
        
        num_norm_avg - int, optional, default 50
            The top 'num_norm_avg' dispersions are averaged to determine the
            normalization factor when calculating the weights. This prevents
            genes with large spatial dispersions from skewing the distribution
            of weights.
        """
        
        self.distance=distance
        
        self.k=k
        if(self.k < 5):
            self.k = 5
        elif(self.k > 100):
            self.k = 100
            
        if(self.k > self.D.shape[0]-1):
            self.k = self.D.shape[0]-2


        numcells = self.D.shape[0]
        
        n_genes = 8000
        if numcells > 3000 and n_genes > 3000:
            n_genes = 3000;
        elif numcells > 2000 and n_genes > 4500:
            n_genes = 4500;
        elif numcells > 1000 and n_genes > 6000:
            n_genes = 6000
        elif n_genes > 8000:        
            n_genes = 8000
        
        npcs = None
        if npcs is None and numcells > 3000:
            npcs = 150
        elif npcs is None and numcells > 2000:
            npcs = 250
        elif npcs is None and numcells > 1000:
            npcs = 350
        elif npcs is None:
            npcs = 500
        
        tinit = time.time()
        
        edm = sp.coo_matrix((numcells,numcells),dtype='i').tolil()
        nums = np.arange(edm.shape[1])
        RINDS = np.random.randint(0,numcells,(self.k-1)*numcells).reshape((numcells,(self.k-1)))
        RINDS = np.hstack((nums[:,None],RINDS))

        edm[np.tile(np.arange(RINDS.shape[0])[:,None],(1,RINDS.shape[1])).flatten(),RINDS.flatten()]=1
        edm=edm.tocsr()
        
        print('RUNNING SAM')         
            
        _, W = self.dispersion_ranking_NN(
            edm, num_norm_avg=1)
               
        old = np.zeros(W.size)
        new = W
        
        i = 0
        err = ((new-old)**2).mean()**0.5

        if max_iter < 5:
            max_iter = 5
            
        nnas=num_norm_avg
        
        
        while (i < max_iter and err > stopping_condition):
        
            conv = err
            if(verbose):
                print('Iteration: ' + str(i) + ', Convergence: ' + str(conv))

            i += 1
            old = new
                        
            W = self.calculate_nnm(W,n_genes,preprocessing,npcs,numcells,nnas)                        
            new = W
            err = ((new-old)**2).mean()**0.5

        self.analysis_performed = True

        if(projection == 'tsne'):
            print('Computing the t-SNE embedding...')
            self.run_tsne()
        elif(projection == 'umap'):
            print('Computing the UMAP embedding...')
            self.run_umap()


        self.datalog = self.raw_data[np.in1d(self.all_cell_names,self.cell_names),:].copy()
        self.datalog.data = np.log2(self.datalog.data+1)
        D_avg=sp.hstack((self.D_avg,sp.coo_matrix((self.cell_names.size,self.all_gene_names.size-self.gene_names.size)))).tocsr()
        
        _,i2 = np.unique(self.all_gene_names,return_inverse=True)
        
        concat = np.append(self.gene_names,self.all_gene_names[np.in1d(self.all_gene_names,self.gene_names,invert=True)])   
        W = np.append(self.weights,np.zeros(self.all_gene_names.size-self.gene_names.size))
        idxs = np.argsort(concat)[i2]
        D_avg = D_avg[:,idxs]
        W = W[idxs]

        self.D_avg_f = self.D_avg
        self.weights_f = self.weights
        self.indinces_f=np.argsort(-self.weights_f)
        
        self.weights=W
        self.D_avg=D_avg

        self.indices = np.argsort(-self.weights)
        self.ranked_genes = self.all_gene_names[self.indices]

        self.corr_bin_genes(number_of_features=1000)

        
        self.output_vars['ranked_gene_indices'] = self.indices
        self.output_vars['ranked_gene_names'] = self.ranked_genes
        self.output_vars['nearest_neighbor_matrix'] = self.nnm_adj
        self.output_vars['gene_weights'] = self.weights
        
        self.adata.obsm['X_pca']=self.wPCA_data
        
        self.adata.uns['neighbors']={}
        self.adata.uns['neighbors']['connectivities']=self.nnm_adj        
        self.adata.var['weights'] = self.weights_f
        self.adata.var['spatial_dispersions'] = self.dispersions
        
        self.adata.layers['X_knn_avg'] = self.D_avg_f
        
        
        elapsed = time.time()-tinit
        if verbose:
            print('Elapsed time: ' + str(elapsed) + ' seconds')
    
    def calculate_nnm(self,W,n_genes,preprocessing,npcs,numcells,num_norm_avg):
        if(n_genes is None):
            gkeep = np.arange(W.size)
        else:
            gkeep = np.sort(np.argsort(-W)[:n_genes])
        
        if preprocessing == 'Normalizer':
            Ds = self.D[:,gkeep].toarray()            
            Ds = Normalizer().fit_transform(Ds)
                       
        elif preprocessing == 'StandardScaler':
            Ds = self.D[:,gkeep].toarray()                
            Ds=StandardScaler(with_mean=True).fit_transform(Ds)            
            Ds[Ds>5]=5
            Ds[Ds<-5]=-5
            
        else:
            Ds = self.D[:,gkeep].toarray()
            
        
        D_sub = Ds*(W[gkeep])
        self.D_sub=D_sub            
        if numcells > 500:
            g_weighted,pca = ut.weighted_PCA(D_sub,npcs=min(npcs,min(self.D.shape)),do_weight=True,solver='auto')
        else:
            g_weighted,pca = ut.weighted_PCA(D_sub,npcs=min(npcs,min(self.D.shape)),do_weight=True,solver='full')
        
        if self.distance=='euclidean':
            g_weighted = Normalizer().fit_transform(g_weighted)
        
        self.wPCA_data = g_weighted
        self.pca = pca
    
        if self.wPCA_data.shape[0] > 8000:
            nnm,dists = ut.nearest_neighbors(g_weighted,n_neighbors=self.k,metric=self.distance)                
            self.knn_dists=dists
            self.knn_indices=nnm
            EDM = sp.coo_matrix((numcells,numcells),dtype='i').tolil()                            
            EDM[np.tile(np.arange(nnm.shape[0])[:,None],(1,nnm.shape[1])).flatten(),nnm.flatten()]=1
            EDM=EDM.tocsr()
        else:                        
            dist = ut.compute_distances(g_weighted, self.distance)
            self.dist=dist
            nnm = ut.dist_to_nn(dist,self.k)                
            EDM = sp.csr_matrix(nnm)
            tx1,tx2 = EDM.nonzero()
            self.knn_indices = tx2.reshape((dist.shape[0],self.k))
            self.knn_dists = dist[np.tile(np.arange(dist.shape[0])[:,None],(1,self.k)).flatten(),self.knn_indices.flatten()].reshape(self.knn_indices.shape)
            
        idx2, W = self.dispersion_ranking_NN(
            EDM, num_norm_avg=num_norm_avg)
        
    
        self.indices = idx2.flatten()
        self.nnm_adj = EDM
        self.weights = W 
        
        return W
        
    def save(self, savename, dirname=None, exc=None):
        """Saves all SAM attributes to a Pickle file.

        Saves all SAM attributes to a Pickle file which can be later loaded
        into an empty SAM object.

        Parameters
        ----------
        savename - string
            The name of the pickle file (not including the file extension) to
            write to.

        dirname - string, optional, default None
            The path/name of the directory in which the Pickle file will be
            saved. If None, the file will be saved to the current working
            directory.

        exc - array-like of strings, optional, default None
            A vector of SAM attributes to exclude from the saved file. Use this
            to exclude bulky objects that do not need to be saved.
            
        """
        self._create_dict(exc)
        if savename[-2:] != '.p':
            savename = savename + '.p'
            
        if(dirname is not None):
            ut.create_folder(dirname+"/")
            f = open(dirname+"/" + savename, 'wb')
        else:
            f = open(savename, 'wb')

        pickle.dump(self.pickle_dict, f)
        f.close()

    def load(self, n):
        """Loads SAM attributes from a Pickle file.

        Loads all SAM attributes from the specified Pickle file into the SAM
        object.

        Parameters
        ----------
        n - string
            The path of the Pickle file.
        """
        f = open(n, 'rb')
        pick_dict = pickle.load(f)
        for i in range(len(pick_dict)):
            self.__dict__[list(pick_dict.keys())[i]
                          ] = pick_dict[list(pick_dict.keys())[i]]
        f.close()

    def _create_dict(self, exc):
        self.pickle_dict = self.__dict__.copy()
        if(exc):
            for i in range(len(exc)):
                try:
                    del self.pickle_dict[exc[i]]
                except NameError:
                    0  

    def plot_top_genes(self, n_genes=5, **kwargs):
        """Plots expression patterns of the top ranked genes.

        Parameters
        ----------
        n_genes - int, optional, default 5
            The number of top ranked genes to display.

        **kwargs -
            All keyword arguments in 'show_gene_expression' and 'scatter'
            are eligible.
        """
        for i in range(n_genes):
            self.show_gene_expression(self.ranked_genes[i], **kwargs)

    def save_figures(self, filename, fig_IDs=None, **kwargs):
        """
        Save figures.
        
        Parameters
        ----------
        filename - str
            Name of output file
            
        fig_IDs - int, numpy.array, list, optional, default None
            A list of open figure IDs or a figure ID that will be saved to a
            pdf/png file respectively.
        
        **kwargs - 
            Extra keyword arguments passed into 'matplotlib.pyplot.savefig'.
            
        """
        if(fig_IDs is not None):
            if(type(fig_IDs) is list):
                savetype = 'pdf'
            else:
                savetype = 'png'
        else:
            savetype = 'pdf'

        if(savetype == 'pdf'):
            from matplotlib.backends.backend_pdf import PdfPages

            if(len(filename.split('.')) == 1):
                filename = filename + '.pdf'
            else:
                filename = '.'.join(filename.split('.')[:-1])+'.pdf'

            pdf = PdfPages(filename)

            if fig_IDs is None:
                figs = [plt.figure(n) for n in plt.get_fignums()]
            else:
                figs = [plt.figure(n) for n in fig_IDs]

            for fig in figs:
                fig.savefig(pdf, format='pdf', **kwargs)
            pdf.close()
        elif(savetype == 'png'):
            plt.figure(fig_IDs).savefig(filename, **kwargs)

    def plot_correlated_groups(self, group=None, n_genes=5, **kwargs):
        """Plots orthogonal expression patterns.

        In the default mode, plots orthogonal gene expression patterns. A
        specific correlated group of genes can be specified to plot gene
        expression patterns within that group.

        Parameters
        ----------
        group - int, optional, default None
            If specified, display the genes within the desired correlated
            group. Otherwise, display the top ranked gene within each distinct
            correlated group.

        n_genes - int, optional, default 5
            The number of top ranked genes to display within a correlated
            group if 'group' is specified.

        **kwargs -
            All keyword arguments in 'show_gene_expression' and 'scatter'
            are eligible.
        """

        if(group is None):
            for i in range(len(self.geneID_groups)):
                self.show_gene_expression(self.geneID_groups[i][0], **kwargs)
        else:
            for i in range(n_genes):
                self.show_gene_expression(self.geneID_groups[group][i], **kwargs)

    def plot_correlated_genes(self, name, n_genes=5,number_of_features=2000, **kwargs):
        """Plots gene expression patterns correlated with the input gene.

        Parameters
        ----------
        name - string
            The name of the gene with respect to which correlated gene
            expression patterns will be displayed.

        n_genes - int, optional, default 5
            The number of top ranked correlated genes to display.

        average_exp - bool, optional, default True
            If True, the plots use the k-nearest-neighbor-averaged expression
            values to smooth out noisy expression patterns and improves
            visualization.

        **kwargs -
            All keyword arguments in 'show_gene_expression' and 'scatter'
            are eligible.
        """
        
        if((self.all_gene_names==name).sum()==0):
            print(
                "Gene not found in the filtered dataset. Note that genes "
                "are case sensitive.")
            return
        sds = self.corr_bin_genes(input_gene=name, number_of_features=number_of_features)
        if (n_genes+1 > sds.size):
            x = sds.size
        else:
            x = n_genes+1
            
        for i in range(1,x):
            self.show_gene_expression(sds[i], **kwargs)
        return sds[1:]

    def corr_bin_genes(self, number_of_features=None, input_gene=None):
        """A (hacky) method for binning groups of genes correlated along the
        SAM manifold.
        
        Parameters
        ----------
        number_of_features - int, optional, default None
            The number of genes to bin. Capped at 5000 due to memory
            considerations.
        
        input_gene - str, optional, default None
            If not None, use this gene as the first seed when growing the
            correlation bins.

        """
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after loading"
                  " the data.")
        else:

            idx2 = np.argsort(-self.weights)[:self.weights[self.weights>0].size]
            
            if(number_of_features is None or number_of_features > idx2.size):
                number_of_features = idx2.size
            
            if number_of_features > 5000:
                number_of_features = 5000;
                
            if(input_gene is not None):
                input_gene = np.where(self.all_gene_names == input_gene)[0]
                if(input_gene.size == 0):
                    print(
                        "Gene note found in the filtered dataset. Note "
                        "that genes are case sensitive.")
                    return
                seeds = [np.array([input_gene])]
                pw_corr = np.corrcoef(
                    self.D_avg[:, idx2[:number_of_features]].T.toarray())
                for i in range(1, number_of_features):
                    flag = False
                    maxd = np.mean(pw_corr[i, :][pw_corr[i, :] > 0])
                    maxi = 0
                    for j in range(len(seeds)):
                        #print(np.where(idx2 == seeds[j][0])[0])
                        if(pw_corr[np.where(idx2 == seeds[j][0])[0], i]
                           > maxd):
                            maxd = pw_corr[np.where(idx2 == seeds[j][0])[0], i]
                            maxi = j
                            flag = True
                    if(not flag):
                        seeds.append(np.array([idx2[i]]))
                    else:
                        seeds[maxi] = np.append(seeds[maxi], idx2[i])

                geneID_groups = []
                for i in range(len(seeds)):
                    geneID_groups.append(self.all_gene_names[seeds[i]])

                return geneID_groups[0]
            else:
                seeds = [np.array([idx2[0]])]
                pw_corr = np.corrcoef(
                    self.D_avg[:, idx2[:number_of_features]].T.toarray())
                for i in range(1, number_of_features):
                    flag = False
                    maxd = np.mean(pw_corr[i, :][pw_corr[i, :] > 0])
                    maxi = 0
                    for j in range(len(seeds)):
                        if(pw_corr[np.where(idx2 == seeds[j][0])[0], i]
                           > maxd):
                            maxd = pw_corr[np.where(idx2 == seeds[j][0])[0], i]
                            maxi = j
                            flag = True
                    if(not flag):
                        seeds.append(np.array([idx2[i]]))
                    else:
                        seeds[maxi] = np.append(seeds[maxi], idx2[i])

                self.geneID_groups = []
                for i in range(len(seeds)):
                    self.geneID_groups.append(
                        self.all_gene_names[seeds[i]])

                return self.geneID_groups
    
    def hdbknn_clustering(self,X=None,k=None,**kwargs):    
        import hdbscan
        if X is None:
            X = self.nnm_adj.toarray()
            save=True
        else:
            save=False
                
        if k is None:
            k = self.k
        
        if k <10:
            k = 10
                        
        hdb = hdbscan.HDBSCAN(metric='correlation',min_cluster_size=k,**kwargs)
    
        cl = hdb.fit_predict(X)
        
        idx0 = np.where(cl != -1)[0]
        idx1 = np.where(cl == -1)[0]
        if idx1.size > 0 and idx0.size > 0:
            xcmap = ut.generate_correlation_map(X[idx0,:],X[idx1,:])
            knn = np.argsort(xcmap.T, axis=1)[:, :k]
            nnm = np.zeros(xcmap.shape).T
            nnm[np.tile(np.arange(knn.shape[0])[:, None],
                        (1, knn.shape[1])).flatten(),
                knn.flatten()] = 1
            nnmc = np.zeros((nnm.shape[0], cl.max() + 1))
            for i in range(cl.max() + 1):
                nnmc[:, i] = nnm[:, cl[idx0] == i].sum(1)
    
            cl[idx1] = np.argmax(nnmc, axis=1)
        
        if save:
            self.output_vars['hdbknn_clusters'] = cl
            self.adata.obs['hdbknn_clusters'] = cl
        else:
            return cl
    
          
    def run_tsne(self, X=None, metric='correlation',**kwargs):
        """Wrapper for sklearn's t-SNE implementation.

        See sklearn for the t-SNE documentation. All arguments are the same
        with the exception that 'metric' is set to 'precomputed' by default,
        implying that this function expects a distance matrix by default.
        """
        if(X is not None):
            dt = man.TSNE(metric=metric,**kwargs).fit_transform(X)
            return dt
        
        elif(not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after "
                  "loading the data.")
        
        else:
            dt = man.TSNE(metric=self.distance,**kwargs).fit_transform(self.wPCA_data)
            self.tsne2d = dt
            self.output_vars['tsne_projection'] = self.tsne2d    
            self.adata.obsm['X_tsne'] = self.tsne2d
            return self.tsne2d.copy()
    
    def run_umap(self, X=None, metric=None, **kwargs):
        """Wrapper for umap-learn.

        See https://github.com/lmcinnes/umap sklearn for the documentation
        and source code.
        """

        import umap as umap

        if metric is None:
            metric = self.distance
        
        if(X is not None):
            umap_obj = umap.UMAP(metric=metric, **kwargs)
            dt = umap_obj.fit_transform(X)            
            return dt
        
        elif (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after "
                  "loading the data.")
        else:
            umap_obj = umap.UMAP(metric=metric, **kwargs)
            self.umap2d = umap_obj.fit_transform(self.wPCA_data)
            self.output_vars['umap_projection'] = self.umap2d    
            self.adata.obsm['X_umap'] = self.umap2d
            return self.umap2d.copy()
        
    def scatter(self, projection=None, c=None, cmap='rainbow', linewidth=0.0,
                edgecolor='k',axes=None,colorbar=True, s=10,**kwargs):
        """Display a scatter plot.

        Displays a scatter plot using the SAM projection or another input
        projection with or without annotations.

        Parameters
        ----------

        projection - ndarray of floats, optional, default None
            An N x 2 matrix, where N is the number of data points. If None,
            use an existing SAM projection (default t-SNE). Can take on values
            'umap' or 'tsne' to specify either the SAM UMAP embedding or
            SAM t-SNE embedding.

        c - ndarray, optional, default None
            Colors for each cell in the scatter plot. Can be a vectory of
            floats or strings for cell annotations.
        
        axes - matplotlib axis, optional, default None
            Plot output to the specified, existing axes. If None, create new
            figure window.
        
        cmap - string, optional, default 'rainbow'
            The colormap to use for the input color values.

        colorbar - bool, optional default True
            If True, display a colorbar indicating which values / annotations
            correspond to which color in the scatter plot.

        Keyword arguments -
            All other keyword arguments that can be passed into
            matplotlib.pyplot.scatter can be used.
        """
        if (not self.analysis_performed and projection is None):
            print("Please run the SAM analysis first using 'run' after loading"
                  " the data.")
        elif (not PLOTTING):
            print("matplotlib not installed!")
        else:
            if(projection == 'umap'):
                try:
                    dt = self.umap2d
                except AttributeError:
                    print('Please create a UMAP projection first.')
                    return
            elif(projection == 'tsne'):
                try:
                    dt = self.tsne2d
                except AttributeError:
                    print('Please create a t-SNE projection first.')
                    return
            elif(projection is None):
                try:
                    dt = self.umap2d
                except AttributeError:
                    try:
                        dt = self.tsne2d
                    except AttributeError:
                        print("Please create either a t-SNE or UMAP projection"
                              "first.")
                        return
            else:
                dt = projection

            if(axes is None):
                plt.figure()
                axes = plt.gca()

            if(c is None):
                plt.scatter(dt[:, 0], dt[:, 1],s=s,linewidth=linewidth,edgecolor=edgecolor, **kwargs)
            else:
                if((type(c[0]) is str or type(c[0]) is np.str_) and
                   (type(c) is np.ndarray or type(c) is list)):
                    i = ut.convert_annotations(c)
                    ui, ai = np.unique(i, return_index=True)
                    cax = axes.scatter(
                        dt[:, 0], dt[:, 1], c=i, cmap=cmap,s=s,linewidth=linewidth,edgecolor=edgecolor, **kwargs)

                    if(colorbar):
                        cbar = plt.colorbar(cax,ax = axes, ticks=ui)
                        cbar.ax.set_yticklabels(c[ai])
                else:
                    if not (type(c) is np.ndarray or type(c) is list):
                        colorbar = False
                    i = c

                    cax = axes.scatter(
                        dt[:, 0], dt[:, 1], c=i, cmap=cmap,s=s,linewidth=linewidth,edgecolor=edgecolor, **kwargs)

                    if(colorbar):
                        plt.colorbar(cax,ax = axes)

    def show_gene_expression(self, gene, average_exp=True, axes=None,**kwargs):
        """Display a gene's expressions.

        Displays a scatter plot using the SAM projection or another input
        projection with a particular gene's expressions overlaid.

        Parameters
        ----------
        gene - string
            a case-sensitive string indicating the gene expression pattern
            to display.

        average_exp - bool, optional, default True
            If True, the plots use the k-nearest-neighbor-averaged expression
            values to smooth out noisy expression patterns and improves
            visualization.
        
        axes - matplotlib axis, optional, default None
            Plot output to the specified, existing axes. If None, create new
            figure window.
            
        **kwargs - all keyword arguments in 'SAM.scatter' are eligible.

        """
        
        if(type(gene) == str or type(gene) == np.str_):
            idx = np.where(self.all_gene_names == gene)[0]
            name = gene
            if(idx.size == 0):
                print(
                    "Gene note found in the filtered dataset. Note that genes "
                    "are case sensitive.")
                return
        else:
            idx = gene
            name = self.gene_names[idx]

        if(average_exp):
            a = self.D_avg[:, idx].toarray().flatten()
        else:
            a = self.datalog[:, idx].toarray().flatten()

        if axes is None:
            plt.figure();
            axes = plt.gca()
            
        self.scatter(c=a,axes=axes, **kwargs)
        axes.set_title(name)

    def sparse_knn(self,D):
        k=self.k
        D1=D.tocoo()
        idr = np.argsort(D1.row)
        D1.row[:]=D1.row[idr]
        D1.col[:]=D1.col[idr]
        D1.data[:]=D1.data[idr]
        
        _,ind = np.unique(D1.row,return_index=True)
        ind = np.append(ind,D1.data.size)
        for i in range(ind.size-1):
            idx = np.argsort(D1.data[ind[i]:ind[i+1]])
            if idx.size > k:
                idx = idx[:-k]
                D1.data[np.arange(ind[i],ind[i+1])[idx]]=0
        D1.eliminate_zeros()
        return D1        
    
    def density_clustering(self,X = None, eps = 1,metric='euclidean',**kwargs):
        from sklearn.cluster import DBSCAN
        
        if X is None:
            X = self.umap2d
            save=True
        else:
            save=False
        
        cl=DBSCAN(eps=eps,metric=metric,**kwargs).fit_predict(X)
        if save:
            self.output_vars['density_clusters'] = cl
            self.adata.obs['density_clusters'] = cl
        else:
            return cl
        
    def louvain_clustering(self,X=None, res=1,method='modularity'):
        """Runs Louvain clustering using the vtraag implementation. Assumes
        that 'louvain' optional dependency is installed.
        
        Parameters
        ----------
        res - float, optional, default 1
            The resolution parameter which tunes the number of clusters Louvain
            finds.       
        
        method - str, optional, default 'modularity'
            Can be 'modularity' or 'significance', which are two different 
            optimizing funcitons in the Louvain algorithm.
        
        """
        
        if X is None:
            X = self.nnm_adj
            save = True
        else:
            if not sp.isspmatrix_csr(X):
                X = sp.csr_matrix(X)
            save = False
            
        import igraph as ig
        import louvain

        adjacency = self.sparse_knn(X.dot(X.T)/self.k).tocsr()
        sources, targets = adjacency.nonzero()
        weights = adjacency[sources, targets]
        if isinstance(weights, np.matrix):
            weights = weights.A1
        g = ig.Graph(directed=True)
        g.add_vertices(adjacency.shape[0])  
        g.add_edges(list(zip(sources, targets)))
        try:
            g.es['weight'] = weights
        except:
            pass

        if method == 'significance':
            cl=louvain.find_partition(g,louvain.SignificanceVertexPartition)
        else:
            cl=louvain.find_partition(g,louvain.RBConfigurationVertexPartition,
                               resolution_parameter=res)
        
        if save:            
            self.output_vars['louvain_clusters'] = np.array(cl.membership)
            self.adata.obs['louvain_clusters'] = np.array(cl.membership)
        else:
            return np.array(cl.membership)

    def identify_marker_genes_rf(self, labels=None,
                              n_genes_subset=3000, svm=False):
        """
        Ranks marker genes for each cluster using a random forest classification
        approach.
        
        Parameters
        ----------
        
        labels - numpy.array or str, optional, default None
            Cluster labels to use for marker gene identification. If None,
            assumes that one of SAM's clustering algorithms has been run. Can
            be a string (i.e. 'louvain_clusters', 'kmeans_clusters', etc) to
            specify specific cluster labels in adata.obs.
        
        n_genes_subset - int, optional, default 3000
            By default, trains the classifier on the top 3000 SAM-weighted genes.           
        
        """
        if(labels is None):
            try:
                lbls = self.output_vars[ut.search_string(np.array(list(
                        self.output_vars.keys())),'_clusters')[0][0]]                
            except AttributeError:
                print("Please generate cluster labels first or set the "
                      "'labels' keyword argument.")
                return
        elif type(labels) is str:
            lbls = self.adata.obs[labels].values.flatten()
        else:
            lbls = labels

        from sklearn.ensemble import RandomForestClassifier

        markers=[]
        lblsu = np.unique(lbls)
        
        X=self.datalog[:,self.indices[:n_genes_subset]].toarray()
        for K in range(lblsu.size):            
            y = np.zeros(lbls.size)

            y[lbls==lblsu[K]]=1                    
            
            clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                  random_state=0)
    
            clf.fit(X, y)

            idx = np.argsort(-clf.feature_importances_)

            markers.append(self.ranked_genes[idx])
    
        markers = np.vstack(markers)                
        self.output_vars['marker_genes_rf'] = markers
        
        return markers
    
    def identify_marker_genes_ratio(self, labels=None):
        """
        Ranks marker genes for each cluster by computing using a SAM-weighted
        expression-ratio approach (works quite well). Marker genes saved in
        'SAM.output_vars' and 'SAM.marker_genes_ratio'.
        
        Parameters
        ----------
        
        labels - numpy.array or str, optional, default None
            Cluster labels to use for marker gene identification. If None,
            assumes that one of SAM's clustering algorithms has been run. Can
            be a string (i.e. 'louvain_clusters', 'kmeans_clusters', etc) to
            specify specific cluster labels in adata.obs.
                
        """        
        if(labels is None):
            try:
                lbls = self.output_vars[ut.search_string(np.array(list(
                        self.output_vars.keys())),'_clusters')[0][0]]                
            except AttributeError:
                print("Please generate cluster labels first or set the "
                      "'labels' keyword argument.")
                return
        elif type(labels) is str:
            lbls = self.adata.obs[labels].values.flatten()
        else:
            lbls = labels
        
        markers = np.zeros(
            (lbls.max()+1, self.all_gene_names.size), dtype=self.all_gene_names.dtype)        
        markers_ratio = np.zeros(markers.shape)
        
        
        s = np.array(self.datalog.sum(0)).flatten()
       
    
        for i in range(lbls.max()+1):
            d = np.array(self.datalog[lbls==i,:].sum(0)).flatten()
            rat = np.zeros(d.size)
            rat[s>0] = d[s>0]**2 / s[s>0] * self.weights[s>0]
            x = np.argsort(-rat)
            markers[i,:]=self.all_gene_names[x[:]]
            markers_ratio[i,:] = rat[x[:]]
        
        self.output_vars['marker_genes_ratio'] = markers
        
        return markers

