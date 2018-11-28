import scipy.sparse as sp
import time
from sklearn.preprocessing import Normalizer, StandardScaler
import pickle
import pandas as pd
import utilities as ut
import sklearn.manifold as man
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


__version__ = '0.2.2'

"""
Copyright 2018, Alexander J. Tarashansky, All rights reserved.
Email: <tarashan@stanford.edu>
"""


"""
TODO: Line-by-line documentation.
"""


class SAM(object):
    """Self-Assembling Manifolds single-cell RNA sequencing analysis tool.

    SAM iteratively rescales the input gene expression matrix to emphasize
    that are spatially variable along the intrinsic manifold of the data.
    It outputs the gene weights, nearest neighbor matrix, distance matrix, and
    a 2D projection.

    Parameters
    ----------
    counts : tuple (scipy.sparse matrix, numpy.array,numpy.array), optional, default None
        A tuple containing sparse data structure of the gene expression counts
        (cells x genes), numpy array of gene IDs, and numpy array of cell IDs.

    annotations : numpy.ndarray, optional, default None
        A Numpy array of cell annotations.

    k : int, optional, default None
        The number of nearest neighbors to identify for each cell. If
        None, k will be automatically be set to the square root of
        the number of cells in the dataset.

    distance : string, optional, default 'correlation'
        The distance metric to use when constructing cell distance
        matrices. Can be any of the distance metrics supported by
        sklearn's 'pdist'.

    Attributes
    ----------
    sparse_data: scipy.sparse matrix
        Sparse data structure of raw gene expression counts.
    
    all_cell_names: numpy.array
        Vector of cell IDs corresponding to rows of sparse_data
    
    all_gene_names: numpy.array
        Vector of gene IDs corresponding to columns of sparse_data

    k: int
        The number of nearest neighbors to identify for each cell
        when constructing the nearest neighbor graph.

    distance: str
        The distance metric used when constructing the cell-to-cell
        distance matrix.
        
    output_vars: dict
        Dictionary of SAM outputs.

    D: scipy.sparse matrix
        The filtered and log-transformed data.

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

    gene_groups: list of numpy.ndarray
        Each element of the list contains a vector of gene indices that are
        correlated with each other.

    geneID_groups: list of numpy.ndarray
        Each element of the list contains a vector of gene IDs that are
        correlated with each other.

    cluster_labels: numpy.array
        Output of 'louvain_clustering'. Denotes assigned cluster label for each cell.
    """

    def __init__(self, counts=None, annotations=None, k=None,
                 distance='correlation'):
        
        if counts is not None:
            self.sparse_data,self.all_gene_names,self.all_cell_names = counts
            self.D = self.sparse_data.copy()
            self.gene_names=self.all_gene_names
            self.cell_names=self.all_cell_names
                  
         
        
        if(annotations is not None):
            self.annotations = annotations
            self.integer_annotations = ut.convert_annotations(self.annotations)

        self.k = k
        self.distance = distance
        self.analysis_performed = False

        self.output_vars = {}
        
        
    def load_sparse_data(self, filename, genename, cellname,
                                   transpose=True, **kwargs):
        
        """Reads the specified sparse data file and cell/gene IDs and stores 
        the data in a scipy.sparse matrix. These files can be obtained by
        running 'load_dense_data_from_file' for the first time.

        
        
        Parameters
        ----------
        filename - string
            The path to the tabular raw expression counts file.
            
        genename - string
            The path to the text file containing gene IDs.
            
        cellname - string
            The path to the text file containing cell IDs.

        sep - string, optional, default ','
            The delimeter used to read the input data table. By default
            assumes the input table is delimited by commas.

        do_filtering - bool, optional, default True
            If True, filters the data with default parameters using
            'filter_data'. Otherwise, loads the data without filtering
            (aside from removing genes with no expression at all).

        transpose - bool, optional, default True
            By default, assumes file is (genes x cells). Set this to False if
            the file has dimensions (cells x genes).

        Keyword arguments
        -----------------

        div : float, default 1
            The factor by which the gene expression will be divided prior to
            log normalization.

        downsample : float, default 0
            The factor by which to randomly downsample the data. If 0, the data
            will not be downsampled.

        genes : array-like of string or int, default None
            A vector of gene names or indices that specifies the genes to keep.
            All other genes will be filtered out. If specified, the usual
            filtering operations do not occur. Gene names are case-sensitive.

        cells : array-like of string or int, default None
            A vector of cell names or indices that specifies the cells to keep.
            All other cells wil lbe filtered out. Cell names are
            case-sensitive.

        min_expression : float, default 1
            The threshold (in log2 TPM) above which a gene is considered
            expressed. Gene expression values less than 'min_expression' are
            set to zero.

        thresh : float, default 0.2
            Keep genes expressed in greater than 'thresh'*100 % of cells and
            less than (1-'thresh')*100 % of cells, where a gene is considered
            expressed if its expression value exceeds 'min_expression'.

        filter_genes : bool, default True
            A convenience parameter. Setting this to False turns off all
            filtering operations.

        """
        import scipy.sparse as sp
        self.sparse_data = sp.load_npz(filename)
        
        if(transpose):
            self.sparse_data = self.sparse_data.T        
            
        self.all_cell_names = np.loadtxt(cellname, dtype='str')
        self.all_gene_names = np.loadtxt(genename, dtype='str')
        
        
        self.filter_sparse_data(**kwargs)                        
    
    def filter_sparse_data(self, div=1, downsample=0, genes=None, cells=None,
                    genes_inc=None,min_expression=1, thresh=0.02, filter_genes=True):
        
        """Log-normalizes and filters the expression data.

        Parameters
        ----------

        div : float, optional, default 1
            The factor by which the gene expression will be divided prior to
            log normalization.

        downsample : float, optional, default 0
            The factor by which to randomly downsample the data. If 0, the
            data will not be downsampled.

        genes : array-like of string or int, optional, default None
            A vector of gene names or indices that specifies the genes to keep.
            All other genes will be filtered out. If specified, the usual
            filtering operations do not occur. Gene names are case-sensitive.

        cells : array-like of string or int, optional, default None
            A vector of cell names or indices that specifies the cells to keep.
            All other cells wil lbe filtered out. Cell names are
            case-sensitive.
        
        genes_inc: array-like of string
            A vector of genes that will be guaranteed to be included after all
            filtering operations (even if they would have otherwise been filtered
            out).
        
        min_expression : float, optional, default 1
            The threshold (in log2 TPM) above which a gene is considered
            expressed. Gene expression values less than 'min_expression' are
            set to zero.

        thresh : float, optional, default 0.2
            Keep genes expressed in greater than 'thresh'*100 % of cells and
            less than (1-'thresh')*100 % of cells, where a gene is considered
            expressed if its expression value exceeds 'min_expression'.

        filter_genes : bool, optional, default True
            Setting this to False turns off all filtering operations aside from
            removing genes with zero expression across all cells.

        """
        if(self.sparse_data is None):
            print('No data loaded')
            return
        
        self.D = self.sparse_data.copy()
        self.D.data = np.log2(self.D.data/div+1)                         
        self.gene_names = self.all_gene_names
        self.cell_names = self.all_cell_names
            
        if(genes is not None):
            genes = np.array(genes)
            
            if str(genes.dtype)[:2] == '<U' or str(genes.dtype) == 'object':
                idx = np.where(
                    (np.in1d(self.gene_names, genes)))[0]
            else:
                idx = genes

            self.D = self.D[:, idx]
            self.gene_names = self.gene_names[idx]
            filter_genes = False
            
            
        if(cells is not None):
            cells = np.array(cells)
            if str(cells.dtype)[:2] == '<U' or str(cells.dtype) == 'object':
                idx2 = np.where(
                    np.in1d(self.cell_names, cells))[0]
            else:
                idx2 = cells

            self.D = self.D[idx2, :]
            self.cell_names = self.cell_names[idx2]


        if downsample > 0:
            numcells = int(self.D.shape[0]/downsample)
            rand_ind= np.random.choice(np.arange(self.D.shape[0]),
                                 size=numcells, replace=False)
            
            self.D = self.D[rand_ind, :]
            self.cell_names = self.cell_names[rand_ind]

        else:
            numcells = self.D.shape[0]


        idx3 = np.where(self.D.data <= min_expression)[0]
        self.D.data[idx3]=0
        self.D.eliminate_zeros()
        
        if(filter_genes):
            a,ct = np.unique(self.D.nonzero()[1],return_counts=True)
            c = np.zeros(self.D.shape[1])
            c[a]=ct
            
            keep = np.where(np.logical_and(c/self.D.shape[0] > thresh,c/self.D.shape[0] < 1-thresh))[0]
        
        else:
            keep = np.where(np.array(self.D.sum(0)).flatten()>0)[0]
            
        
        if(genes_inc is not None):
            keep = np.append(keep, np.where(np.in1d(self.gene_names,np.array(genes_inc)))[0])
            keep = np.unique(keep)       
        
                                
        self.D=self.D.tocsc()
        self.D = self.D[:,keep]
        
        self.gene_names = self.gene_names[keep]
                
        
    def load_dense_data_from_file(self, filename, do_filtering=True, transpose=True,
                            save_sparse_files=True,sep=',', **kwargs):
        """Reads the specified tabular data file and stores the data in a
        scipy.sparse matrix.

        This is a wrapper function that loads the file specified by 'filename'
        and filters the data.

        Parameters
        ----------
        filename - string
            The path to the tabular raw expression counts file.

        sep - string, optional, default ','
            The delimeter used to read the input data table. By default
            assumes the input table is delimited by commas.

        do_filtering - bool, optional, default True
            If True, filters the data with default parameters using
            'filter_data'. Otherwise, loads the data without filtering
            (aside from removing genes with no expression at all).
        
        save_sparse_files - bool, optional, default True
            If True, saves the gene names, cell names, and sparse data structure
            in the same folder as the original data for faster loading in the
            future using 'load_sparse_data'. 
            
        transpose - bool, optional, default True
            By default, assumes file is (genes x cells). Set this to False if
            the file has dimensions (cells x genes).

        Keyword arguments
        -----------------

        div : float, default 1
            The factor by which the gene expression will be divided prior to
            log normalization.

        downsample : float, default 0
            The factor by which to randomly downsample the data. If 0, the data
            will not be downsampled.

        genes : array-like of string or int, default None
            A vector of gene names or indices that specifies the genes to keep.
            All other genes will be filtered out. If specified, the usual
            filtering operations do not occur. Gene names are case-sensitive.

        cells : array-like of string or int, default None
            A vector of cell names or indices that specifies the cells to keep.
            All other cells wil lbe filtered out. Cell names are
            case-sensitive.

        min_expression : float, default 1
            The threshold (in log2 TPM) above which a gene is considered
            expressed. Gene expression values less than 'min_expression' are
            set to zero.

        thresh : float, default 0.2
            Keep genes expressed in greater than 'thresh'*100 % of cells and
            less than (1-'thresh')*100 % of cells, where a gene is considered
            expressed if its expression value exceeds 'min_expression'.

        filter_genes : bool, default True
            A convenience parameter. Setting this to False turns off all
            filtering operations.

        """
        df = pd.read_csv(filename, sep=sep, index_col=0)
        if(transpose):
            dataset = df.T
        else:
            dataset = df
        
        self.sparse_data = sp.csr_matrix(dataset.values)
        self.all_cell_names = np.array(list(dataset.index.values))
        self.all_gene_names = np.array(list(dataset.columns.values))
        
        self.filter_sparse_data(**kwargs)        
        
        if(save_sparse_files):
            new_sparse_file = filename.split('/')[-1].split('.')[0]        
            path = filename[:filename.find(filename.split('/')[-1])]

            np.savetxt(path+new_sparse_file+'_cells.txt',self.all_cell_names,fmt='%s')
            np.savetxt(path+new_sparse_file+'_genes.txt',self.all_gene_names,fmt='%s')
            sp.save_npz(path+new_sparse_file+'_sparse.npz',self.sparse_data.T)
            
           
    def load_annotations(self, aname,**kwargs):
        """Loads cell annotations.

        Loads the cell annoations specified by the 'aname' path.
        
        Keyword arguments:
            See numpy.loadtxt documentation.

        """
        self.annotations = np.loadtxt(aname,dtype='str',**kwargs)    
        self.integer_annotations = ut.convert_annotations(self.annotations)
    
    
    
    def dispersion_ranking_NN(self, nnm, num_norm_avg=50):
        """Computes the spatial dispersion factors for each gene.      

        Parameters
        ----------
        nnm - scipy.sparse, float
            Square cell-to-cell nearest-neighbor matrix.

        num_norm_avg - int, optional, default 50
            The top 'num_norm_avg' dispersions are averaged to determine the
            normalization factor when calculating the weights.

        Returns:
        -------
        indices - ndarray, int
            The indices corresponding to the gene weights sorted in decreasing
            order.

        weights - ndarray, float
            The vector of gene weights.
        """
        
        D_avg = ((nnm/self.k).dot(self.D))
        
        self.D_avg = D_avg.copy()

        mu = D_avg.mean(0)
        Ex2=np.square(mu) 
        D_avg.data=D_avg.data**2
        Ex1=D_avg.mean(0)
        var = np.array(Ex1-Ex2).flatten()
        mu = np.array(mu).flatten()
        
        dispersions = np.zeros(var.size)
        dispersions[mu>0] = var[mu>0]/mu[mu>0]
        
        ma = np.sort(dispersions)[-num_norm_avg:].mean()
        dispersions[dispersions >= ma] = ma
        weights = ((dispersions/dispersions.max())**0.5).flatten()        

        indices = np.argsort(-weights)
        
        return indices, weights

    def run(self,
            max_iter=15,
            stopping_condition=1e-3,
            verbose=True,
            projection=None,
            n_genes=2000,
            npcs=150,
            num_norm_avg=50,
            weight_PCs=True,
            final_PC=True):
        """Runs the Self-Assembling Manifold algorithm.

        Parameters
        ----------
        max_iter - int, optional, default 15
            The maximum number of iterations SAM will run.

        stopping_condition - float, optional, default 1e-3
            The convergence threshold for the error between adjacent cell
            distance matrices.

        verbose - bool, optional, default True
            If True, the iteration number and convergence score will be
            displayed.

        projection - str, optional, default None
            If 'tsne', generates a t-SNE embedding. If 'umap', generates a UMAP
            embedding. Otherwise, no embedding will be generated.

        npcs - int, optional, default 150
            Determines the number of weighted principal
            components to take. If None, all principal components will be
            selected. For large datasets (>5000 cells), we recommend 'npcs' to
            be lowered (e.g. npcs = 500) if runtime is an issue. Otherwise,
            selecting all principal components should be fine.

        n_genes - int, optional, default None
            Improve runtime by selecting only the top 'n_genes' weighted genes
            when computing PCA. If None, use all genes.

        num_norm_avg - int, optional, default 50
            The top 'num_norm_avg' dispersions are averaged to determine the
            normalization factor when calculating the weights.
        """
        if(not self.k):
            self.k = 15

        if(self.k < 5):
            self.k = 5
        elif(self.k > 100):
            self.k = 100

        if(n_genes is not None):
            if(n_genes < 2*num_norm_avg):
                n_genes = 2*num_norm_avg

        if(self.k > self.D.shape[0]-1):
            print("Warning: chosen k exceeds the number of cells")
            self.k = self.D.shape[0]-2


        numcells = self.D.shape[0]
        tinit = time.time()
        
        edm = sp.coo_matrix((numcells,numcells),dtype='i').tolil()
        nums = np.arange(edm.shape[1])
        RINDS = np.random.randint(0,numcells,(self.k-1)*numcells).reshape((numcells,(self.k-1)))
        RINDS = np.hstack((nums[:,None],RINDS))

        edm[np.tile(np.arange(RINDS.shape[0])[:,None],(1,RINDS.shape[1])).flatten(),RINDS.flatten()]=1
        edm=edm.tocsr()
        
        print('RUNNING SAM')         
            
        _, W = self.dispersion_ranking_NN(
            edm, num_norm_avg=num_norm_avg)
        
        old = np.zeros(W.size)
        new = W
        
        i = 0
        err = ((new-old)**2).mean()**0.5


        while (err > stopping_condition and i < max_iter):
        
            conv = err
            if(verbose):
                print('Iteration: ' + str(i) + ', Convergence: ' + str(conv))

            i += 1
            old = new
            if(n_genes is None):
                gkeep = np.arange(W.size)
            else:
                gkeep = np.sort(np.argsort(-W)[:n_genes])
            
            Ds=StandardScaler(with_mean=True).fit_transform(self.D[:,gkeep].toarray())
            Ds[Ds>5]=5
            Ds[Ds<-5]=-5
            D_sub = Ds*(W[gkeep])
            g_weighted,pca = ut.weighted_PCA(D_sub,npcs=npcs)
            
            self.wPCA_data = g_weighted

            if self.wPCA_data.shape[0] > 8000:
                nnm,_ = ut.nearest_neighbors(g_weighted,n_neighbors=self.k,metric=self.distance)                
                EDM = sp.coo_matrix((numcells,numcells),dtype='i').tolil()                            
                EDM[np.tile(np.arange(nnm.shape[0])[:,None],(1,nnm.shape[1])).flatten(),nnm.flatten()]=1
                EDM=EDM.tocsr()
            else:                        
                dist = ut.compute_distances(g_weighted, self.distance)
                EDM = sp.csr_matrix(ut.dist_to_nn(dist,self.k))
                
            idx2, W = self.dispersion_ranking_NN(
                EDM, num_norm_avg=num_norm_avg)
            

            self.indices = idx2.flatten()
            self.nnm_adj = EDM
            self.weights = W
            new = W
            err = ((new-old)**2).mean()**0.5


        self.ranked_genes = self.gene_names[self.indices]

        self.output_vars['ranked_gene_indices'] = self.indices
        self.output_vars['ranked_gene_names'] = self.ranked_genes
        self.output_vars['nearest_neighbor_matrix'] = self.nnm_adj
        self.output_vars['gene_weights'] = self.weights

        self.analysis_performed = True
        
        if(projection is 'tsne'):
            print('Computing the t-SNE embedding...')
            self.run_tsne()
        elif(projection is 'umap'):
            print('Computing the UMAP embedding...')
            self.run_umap()

        self.corr_bin_genes(number_of_features=2000)

        elapsed = time.time()-tinit

        print('Elapsed time: ' + str(elapsed) + ' seconds')

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
            A vector of SAM attributes to exclude from the saved file.
        """
        self._create_dict(exc)

        if(dirname is not None):
            ut.create_folder(dirname+"/")
            f = open(dirname+"/" + savename + ".p", 'wb')
        else:
            f = open(savename + ".p", 'wb')

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

    def save_marker_genes_to_pdf(self, filename, **kwargs):
        nclusts = self.cluster_labels.max()+1
        lbls = np.tile(np.arange(nclusts)[
                       :, None], (1, self.marker_genes.shape[1]))
        lbls = lbls.flatten()
        lbls_colors = np.zeros_like(self.cluster_labels)

        try:
            plt.ioff()
            for i, gene in enumerate(self.marker_genes.flatten()):
                lbls_colors[:] = 0
                lbls_colors[self.cluster_labels == lbls[i]] = 1
                plt.figure(figsize=(12, 5))
                ax1 = plt.subplot(121)
                self.show_gene_expression(gene, axes=ax1, **kwargs)
                ax2 = plt.subplot(122)
                self.scatter(c=lbls_colors, colorbar=False, axes=ax2, **kwargs)
                plt.set_cmap('rainbow')
                plt.title('Cluster: ' + str(lbls[i]))

            self.save_figures(filename)
            plt.close('all')
            plt.ion()

        except AttributeError:
            print("Please run 'identify_marker_genes' first.")

    def save_figures(self, filename, fig_IDs=None, **kwargs):

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
        self.corr_bin_genes(number_of_features=2000)

        if(group is None):
            for i in range(len(self.gene_groups)):
                self.show_gene_expression(self.gene_names[self.gene_groups[i][0]], **kwargs)
        else:
            for i in range(n_genes):
                self.show_gene_expression(self.gene_names[self.gene_groups[group][i]], **kwargs)

    def plot_correlated_genes(self, name, n_genes=5, **kwargs):
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
        name = np.where(self.gene_names == name)[0]
        if(name.size == 0):
            print(
                "Gene note found in the filtered dataset. Note that genes "
                "are case sensitive.")
            return
        sds, _ = self.corr_bin_genes(input_gene=name, number_of_features=2000)

        for i in range(1, n_genes+1):
            self.show_gene_expression(self.gene_names[sds[0][i]], **kwargs)
        return self.gene_names[sds[0][1:]]

    def corr_bin_genes(self, number_of_features=None, input_gene=None):
        """A (hacky) method for binning groups of correlated genes.

        """
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after loading"
                  " the data.")
        else:

            idx2 = np.argsort(-self.weights)

            if(number_of_features is None or number_of_features > idx2.size):
                number_of_features = idx2.size

            if(input_gene is not None):
                if(type(input_gene) is str):
                    input_gene = np.where(self.gene_names == input_gene)[0]
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
                    geneID_groups.append(self.gene_names[seeds[i]])

                return seeds, geneID_groups
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

                self.gene_groups = seeds
                self.geneID_groups = []
                for i in range(len(self.gene_groups)):
                    self.geneID_groups.append(
                        self.gene_names[self.gene_groups[i]])

                return seeds
    
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
            dt = man.TSNE(metric=metric,**kwargs).fit_transform(self.wPCA_data)
            self.tsne2d = dt
            self.output_vars['tsne_projection'] = self.tsne2d        
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
            return self.umap2d.copy()
        
    def scatter(self, projection=None, c=None, cmap='rainbow', linewidth=0.1,
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

        cmap - string, optional, default 'rainbow'
            The colormap to use for the input color values.

        new_figure - bool, optional, default True
            If True, creates a new figure. Otherwise, outputs the scatter plot
            to currently active matplotlib axes.

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
            if(projection is 'umap'):
                try:
                    dt = self.umap2d
                except AttributeError:
                    print('Please create a UMAP projection first.')
                    return
            elif(projection is 'tsne'):
                try:
                    dt = self.tsne2d
                except AttributeError:
                    print('Please create a t-SNE projection first.')
                    return
            elif(projection is None):
                try:
                    dt = self.tsne2d
                except AttributeError:
                    try:
                        dt = self.umap2d
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
                        cbar = plt.colorbar(cax, ticks=ui)
                        cbar.ax.set_yticklabels(c[ai])
                else:
                    if not (type(c) is np.ndarray or type(c) is list):
                        colorbar = False
                    i = c

                    cax = axes.scatter(
                        dt[:, 0], dt[:, 1], c=i, cmap=cmap,s=s,linewidth=linewidth,edgecolor=edgecolor, **kwargs)

                    if(colorbar):
                        plt.colorbar(cax)

    def show_gene_expression(self, gene, average_exp=True, **kwargs):
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

        **kwargs - keyword arguments in 'scatter' are eligible

        """
        
        if(type(gene) == str or type(gene) == np.str_):
            idx = np.where(self.gene_names == gene)[0]
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
            a = self.D[:, idx].toarray().flatten()

        self.scatter(c=a, **kwargs)
        plt.title(name)


    def louvain_clustering(self, res=1):
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after "
                  "loading the data.")
        else:
            import anndata
            import scanpy.api as sc
            adata = anndata.AnnData(self.D, var={'genes': self.gene_names},
                                    obs={'cells': self.cell_names})
            adata.obsm['X_pca'] = self.wPCA_data
            sc.pp.neighbors(adata, n_neighbors=self.k, metric='correlation',
                            method='umap')
            sc.tl.louvain(adata, resolution=res)
            self.cluster_labels = adata.obs['louvain'].values.astype('int')
            self.output_vars['louvain_cluster_labels'] = self.cluster_labels

    def identify_marker_genes(self, n_genes_per_cluster=10, labels=None,
                              n_genes_subset=2000, svm=True):
        if(labels is None):
            try:
                lbls = self.cluster_labels
            except AttributeError:
                print("Please generate cluster labels first or set the "
                      "'labels' keyword argument.")
                return
        else:
            lbls = labels

        if(not svm):
            import sklearn.linear_model
            obj = sklearn.linear_model.LogisticRegression(
                solver='liblinear', multi_class='auto')
        else:
            import sklearn.svm
            obj = sklearn.svm.LinearSVC()

        rawD = self.dataset[list(self.ranked_genes[:n_genes_subset])]
        obj.fit(rawD.values, lbls)
        idx = np.argsort(-(obj.coef_), axis=1)

        markers = np.zeros(
            (idx.shape[0], n_genes_per_cluster), dtype=self.gene_names.dtype)
        for i in range(idx.shape[0]):
            markers[i, :] = rawD.columns[idx[i, :n_genes_per_cluster]]

        self.marker_genes = markers
        self.output_vars['marker_genes'] = self.marker_genes

    def identify_marker_genes2(self, n_genes_per_cluster=10, labels=None):
        if(labels is None):
            try:
                lbls = self.cluster_labels
            except AttributeError:
                print("Please generate cluster labels first or set the "
                      "'labels' keyword argument.")
                return
        else:
            lbls = labels
        
        markers = np.zeros(
            (lbls.max()+1, n_genes_per_cluster), dtype=self.gene_names.dtype)        
        for i in range(lbls.max()+1):
            d = self.D[lbls==i,:]
            x = np.argsort(-d.sum(0)**2 / self.D.sum(0))
            markers[i,:]=self.gene_names[x[:n_genes_per_cluster]]

        self.marker_genes = markers
        self.output_vars['marker_genes'] = self.marker_genes
