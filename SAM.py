"""
Copyright 2018, Alexander J. Tarashansky, All rights reserved.
Email: <tarashan@stanford.edu>
"""




"""
TODO: Line-by-line documentation.
"""

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import Normalizer
import pickle
import pandas as pd
import numpy as np
import utilities as ut
import sklearn.manifold as man

class SAM(object):
    """Self-Assembling Manifolds single-cell RNA sequencing analysis tool.

    SAM iteratively rescales the input gene expression matrix to emphasize
    that are spatially variable along the intrinsic manifold of the data.
    It outputs the gene weights, nearest neighbor matrix, distance matrix, and
    a 2D projection.

    Parameters
    ----------
    filename : string
        The path to the data file. 

    ann_name : string, optional, default None
        The path to the cell annotation file, should one exist.

    k : int, optional, default None
        The number of nearest neighbors to identify for each cell. If None,
        k will be automatically be set to the square root of the number of
        cells in the dataset
        
    distance : string, optional, default 'correlation'
        The distance metric to use when constructing cell distance matrices.
        Can be any of the distance metrics supported by sklearn's 'pdist'.
        
    Attributes
    ----------
    filename: The path to the data file.

    ann_name: The path to the cell annotations file (optional).

    k: The number of nearest neighbors to identify for each cell when constructing the nearest neighbor graph.

    distance: The distance metric used when constructing the cell-to-cell distance matrix.

    dataset: A Pandas DataFrame containing the original input data (cells x genes).

    filtered_dataset: A Pandas DataFrame containing the filtered data (cells x genes).

    num_expressed_genes: The number of expressed genes in each cell.

    D: The numpy array version of 'filtered_dataset' (equivalent to filtered_dataset.values).

    gene_names: A vector of the gene names (equivalent to filtered_dataset.columns).

    cell_names: A vector of the cell names (equivalent to filtered_dataset.index).

    ann: A vector of cell annotations if they were provided.

    ann_int: A vector of cell annotations converted to integers.

    D_avg: The k-nearest-neighbor-averaged expression data.

    weighted_data: The rescaled expression data.

    wPCA_data: The weighted PCA matrix.

    pca: The sklearn pca object.

    dist: The cell-to-cell distance matrix.

    indices: Indices sorting the genes according to the weights in descending order (i.e. indices[0] is the index of the gene with the largest weight).

    nnm_adj: The nearest neighbor adjacency matrix.

    weights: A vector of weights for each gene.

    ranked_genes: The ranked list of genes, i.e. sam.gene_names[sam.indices].

    dt: The t-SNE embedding.

    gene_groups: A list of numpy arrays, where each element of the list contains a vector of gene indices that are correlated with each other.

    geneID_groups: A list of numpy arrays, where each element of the list contains a vector of gene IDs that are correlated with each other.
    
    """
    
    def __init__(self,filename,ann_name=None,k=None,distance='correlation'):
        self.filename=filename
        self.ann_name=ann_name
        self.k=k
        self.distance=distance
        self.analysis_performed=False
  
    def load_data(self,sep=',',**kwargs):
        """Reads the specified file and stores the data in a Pandas DataFrame.

        This is a wrapper function that loads the file specified by 'filename', 
        filters the data, and loads cell annotations if 'ann_name' is not None.
        
        Parameters
        ----------
        sep - string, optional, default ','
            The delimeter used to read the input data table. By default assumes the input table is delimited by commas.
        
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
        df = pd.read_csv(self.filename,sep=sep, index_col=0) 
        self.dataset=df.T  
        self.filter_data(**kwargs)
        self.load_annotations()        
    
    def filter_data(self,div=1,downsample=0,genes=None,cells=None,min_expression=1,thresh=0.02,filter_genes=True):              
        """Log-normalizes and filters the expression data.
        
        Parameters
        ----------
        
        div : float, optional, default 1
            The factor by which the gene expression will be divided prior to 
            log normalization.
        
        downsample : float, optional, default 0
            The factor by which to randomly downsample the data. If 0, the data
            will not be downsampled.
        
        genes : array-like of string or int, optional, default None
            A vector of gene names or indices that specifies the genes to keep.
            All other genes will be filtered out. If specified, the usual
            filtering operations do not occur. Gene names are case-sensitive.
        
        cells : array-like of string or int, optional, default None
            A vector of cell names or indices that specifies the cells to keep.
            All other cells wil lbe filtered out. Cell names are 
            case-sensitive.
            
        min_expression : float, optional, default 1
            The threshold (in log2 TPM) above which a gene is considered 
            expressed. Gene expression values less than 'min_expression' are 
            set to zero.
        
        thresh : float, optional, default 0.2
            Keep genes expressed in greater than 'thresh'*100 % of cells and 
            less than (1-'thresh')*100 % of cells, where a gene is considered
            expressed if its expression value exceeds 'min_expression'.
            
        filter_genes : bool, optional, default True
            A convenience parameter. Setting this to False turns off all
            filtering operations. 
       
        """
        self.filtered_dataset=np.log2(self.dataset/div+1)
        
        if(genes is not None):
            genes=np.array(genes)
            
            if str(genes.dtype)[:2]=='<U' or str(genes.dtype)=='object':    
                idx=np.where((np.in1d(self.filtered_dataset.columns.values,genes)))[0]
            else:
                idx = genes
                
            self.filtered_dataset=self.filtered_dataset.iloc[:,idx]   
            
            filter_genes = False
        
        if(cells is not None):
            cells=np.array(cells)
            if str(cells.dtype)[:2]=='<U' or str(cells.dtype)=='object':    
                idx2=np.where(np.in1d(self.filtered_dataset.index.values,cells))[0]
            else:
                idx2 = cells
            
            self.filtered_dataset=self.filtered_dataset.iloc[idx2,:]     
                        
        
        if downsample > 0:
            numcells = int(self.filtered_dataset.shape[0]/downsample)
            self.filtered_dataset=self.filtered_dataset.iloc[np.random.choice(np.arange(self.filtered_dataset.shape[0]),size=numcells,replace=False),:]
            
        else:            
            numcells=self.filtered_dataset.shape[0] 



        temp_data=np.array(self.filtered_dataset)
                
        temp_data[temp_data<=min_expression]=0; 
        temp_data[temp_data>0]=1      
                
        self.num_expressed_genes=temp_data.sum(1)
            
        if(filter_genes):
            keep=np.where(np.logical_and(np.sum(temp_data,axis=0)/numcells>thresh,
                                np.sum(temp_data,axis=0)/numcells<=1-thresh))[0]            
        else:
            keep =np.where(temp_data.sum(0)>0)[0]
            
        self.filtered_dataset = self.filtered_dataset.iloc[:,keep]        
        self.filtered_dataset[self.filtered_dataset<=min_expression]=0        

        self.D=self.filtered_dataset.values

        self.gene_names=np.array(list(self.filtered_dataset.columns.values))
        self.cell_names=np.array(list(self.filtered_dataset.index.values))
     
    def load_annotations(self,ann_name=None):   
        """Loads cell annotations.
        
        Loads the cell annoations specified by 'ann_name' during the creation
        of the SAM object.

        """        
        aname = self.ann_name        

        if(aname):            
            ann = pd.read_csv(aname,header=None)                
            if(ann.size!=self.dataset.shape[0]):
                ann = pd.read_csv(aname,index_col=0,header=None)
            if(ann.size!=self.dataset.shape[0]):
                ann = pd.read_csv(aname,index_col=0)
            if(ann.size!=self.dataset.shape[0]):
                ann = pd.read_csv(aname)
                                       
            if(ann.size!=self.filtered_dataset.shape[0]):
                ann=ann.values.flatten()[np.where(np.in1d(self.dataset.index.values,self.filtered_dataset.index.values))[0]]
            else:
                ann=ann.values.flatten()
            
            self.ann=ann
            self.ann_int=ut.convert_annotations(self.ann)  
            
    def dispersion_ranking_NN(self,dist):
        """Computes the spatial dispersion factors for each gene.
        
        Given a cell distance matrix, this function calculates the k-nearest
        neighbor adjacency matrix, and performs k-nearest-neighbor averaging
        of the expression data. From the averaged expression data, the Fano 
        factor (variance/mean) for each gene is computed. These factors are
        square rooted and then min-max normalized to generate the
        gene weights, from which gene rankings are calculated.
        
        
        Loads the cell annoations specified by 'ann_name' during the creation
        of the SAM object.
        
        Parameters
        ----------
        dist - ndarray, float
            Square cell-to-cell distance matrix.
        
        
        Returns:
        -------
        indices - ndarray, int
            The indices corresponding to the gene weights sorted in decreasing
            order.
            
        weights - ndarray, float
            The vector of gene weights.
        
        nnm - ndarray, int
            The square k-nearest-neighbor directed adjacency matrix.
        
        D_avg - ndarray, float
            The k-nearest-neighbor-averaged expression data.
        
        """           
        nnm=ut.dist_to_nn(dist,self.k)     
        
        D_avg=nnm.dot(self.D)/np.sum(nnm,axis=1).reshape(self.D.shape[0],1)
        
        dispersions=D_avg.var(0)/D_avg.mean(0)
        
        weights = ut.normalizer(dispersions**0.5)
        
        indices=np.argsort(-weights)

        self.D_avg=D_avg

        return indices,weights,nnm,D_avg


    def run(self,max_iter=15,stopping_condition=1e-4,verbose=True,projection=True,npcs=None):
        """Runs the Self-Assembling Manifold algorithm.
        
        Parameters
        ----------
        max_iter - int, optional, default 15
            The maximum number of iterations SAM will run.
        
        stopping_condition - float, optional, default 1e-4
            The convergence threshold for the error between adjacent cell 
            distance matrices.
            
        verbose - bool, optional, default True
            If True, the iteration number and convergence score will be
            displayed
            
        projection - bool, optional, default False
            If True, performs t-SNE embedding on the cell-cell distance 
	    matrix. If False, no 2D projection will be generated. 
            
        npcs - int, optional, default None
            Determines the number of weighted principal 
            components to take. If None, all principal components will be 
            selected. For large datasets (>5000 cells), we recommend 'npcs' to 
            be lowered (e.g. npcs = 500) if runtime is an issue. Otherwise,
            selecting all principal components should be fine.
        """  
        
        if(not self.k):            
            self.k = int(self.D.shape[0]**0.5)
        
        if(self.k<5):
            self.k=5            
        elif(self.k>100):
            self.k = 100
        
        if(self.k > self.D.shape[0]-1):
            print("Warning: chosen k exceeds the number of cells")
            self.k = self.D.shape[0]-2
        
        
        print('RUNNING ' + self.filename)        
        
        numcells=self.D.shape[0]
        tinit=time.time()


        dist=np.random.rand(numcells,numcells)*2
        dist[np.arange(numcells),np.arange(numcells)]=0


        _,dispersions,edm,_=self.dispersion_ranking_NN(dist)
        
        W=dispersions.reshape((1,self.D.shape[1]))

        old=dist
        
        new=np.random.rand(numcells,numcells)*2        
        new[np.arange(numcells),np.arange(numcells)]=0        

        i=0 
        
        err=ut.distance_matrix_error(new,old)

        while (err > stopping_condition and i < max_iter):

            conv=err
            if(verbose):
                print('Iteration: ' + str(i) + ', Convergence: ' + str(conv))

            i+=1 
            old=new
            
            weighted_data = self.D*W.flatten()[None,:]
            self.weighted_data=weighted_data

            g_weighted,pca=ut.weighted_PCA(Normalizer().fit_transform(weighted_data))

            self.wPCA_data=g_weighted
            self.pca=pca

            dist=ut.compute_distances(g_weighted,self.distance)
            idx2,dispersions,EDM,_=self.dispersion_ranking_NN(dist)

            W = dispersions.reshape((1,self.D.shape[1]))

            self.dist=dist
            self.indices=idx2.flatten()
            self.nnm_adj=EDM
            self.weights=W.flatten()
            new=dist
            
            err=ut.distance_matrix_error(new,old)

        self.ranked_genes=self.gene_names[self.indices]
        
        
        if(projection):
            print('Computing the t-SNE embedding...')
            self.run_tsne()
        else:
            self.dt = None

        self.analysis_performed=True

        self.corr_bin_genes(number_of_features=2000);
        
        elapsed=time.time()-tinit
        
        print('Elapsed time: ' + str(elapsed) + ' seconds')
        
        
    def save(self,savename,dirname=None,exc=None):
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
            f = open(dirname+"/" + savename +".p", 'wb')
        else:
            f = open(savename +".p", 'wb')
            
        pickle.dump(self.pickle_dict, f)
        f.close()
    
    def load(self,n):
        """Loads SAM attributes from a Pickle file.
        
        Loads all SAM attributes from the specified Pickle file into the SAM
        object.
        
        Parameters
        ----------
        n - string
            The path of the Pickle file.
        """        
        f = open(n, 'rb')        
        pick_dict=pickle.load(f)
        for i in range(len(pick_dict)):
            self.__dict__[list(pick_dict.keys())[i]]=pick_dict[list(pick_dict.keys())[i]]
        f.close()

    def _create_dict(self,exc):
        self.pickle_dict = self.__dict__.copy()
        if(exc):
            for i in range(len(exc)):
                try:
                    del self.pickle_dict[exc[i]]
                except:
                    0; # do nothing
    
    def plot_top_genes(self,n_genes=5,average_exp=True):
        """Plots expression patterns of the top ranked genes.
        
        Parameters
        ----------
        n_genes - int, optional, default 5
            The number of top ranked genes to display.
        
        average_exp - bool, optional, default True
            If True, the plots use the k-nearest-neighbor-averaged expression 
            values to smooth out noisy expression patterns and improves
            visualization.        
        """
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after loading the data.")
        elif (self.dt is None):
            print("Please create a 2D projection first using 'run_tsne'. ")
        else:
            for i in range(n_genes):
                self.show_gene_expression(self.indices[i],average_exp=average_exp)
       


    def plot_correlated_groups(self,group=None,n_genes=5, average_exp=True):
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
        
        average_exp - bool, optional, default True
            If True, the plots use the k-nearest-neighbor-averaged expression 
            values to smooth out noisy expression patterns and improves
            visualization.        
        """        
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after loading the data.")
        elif (self.dt is None):
            print("Please create a 2D projection first using 'run_tsne'. ")
        else:
            self.corr_bin_genes(number_of_features=2000);
                
            if(group is None):
                for i in range(len(self.gene_groups)):
                    self.show_gene_expression(self.gene_groups[i][0],average_exp=average_exp)
            else:
                for i in range(n_genes):
                    self.show_gene_expression(self.gene_groups[group][i],average_exp=average_exp)
             
               

    def plot_correlated_genes(self,name,n_genes=5,average_exp=True):       
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
        """               
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after loading the data.")
        elif (self.dt is None):
            print("Please create a 2D projection first using 'run_tsne'. ")
        else:        
            name=np.where(self.gene_names==name)[0]
            if(name.size==0):
                print("Gene note found in the filtered dataset. Note that genes are case sensitive.")
                return  
            sds,_=self.corr_bin_genes(input_gene=name,number_of_features=2000)

            for i in range(1,n_genes+1):
                self.show_gene_expression(sds[0][i],average_exp=average_exp)
            return self.gene_names[sds[0][1:]]


    def corr_bin_genes(self,number_of_features=None,input_gene=None):
        """A (hacky) method for binning groups of correlated genes.        
        
        """
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after loading the data.")
        else:
            
            idx2=np.argsort(-self.weights)

            if(number_of_features is None or number_of_features>idx2.size):
                number_of_features=idx2.size

            if(input_gene is not None):
                if(type(input_gene) is str):
                    input_gene=np.where(self.gene_names==input_gene)[0]
                    if(input_gene.size==0):
                        print("Gene note found in the filtered dataset. Note that genes are case sensitive.")
                        return
                seeds=[np.array([input_gene])]
                pw_corr=np.corrcoef(self.D_avg[:,idx2[:number_of_features]].T)
                for i in range(1,number_of_features):
                    flag = False
                    maxd=np.mean(pw_corr[i,:][pw_corr[i,:]>0])
                    maxi=0
                    for j in range(len(seeds)):
                        if(pw_corr[np.where(idx2==seeds[j][0])[0],i] > maxd):
                            maxd=pw_corr[np.where(idx2==seeds[j][0])[0],i]
                            maxi=j
                            flag=True
                    if(not flag):
                        seeds.append(np.array([idx2[i]]))
                    else:
                        seeds[maxi]=np.append(seeds[maxi],idx2[i])

                geneID_groups=[]
                for i in range(len(seeds)):
                    geneID_groups.append(self.gene_names[seeds[i]])
                
                return seeds,geneID_groups
            else:
                seeds=[np.array([idx2[0]])]
                pw_corr=np.corrcoef(self.D_avg[:,idx2[:number_of_features]].T)
                for i in range(1,number_of_features):
                    flag = False
                    maxd=np.mean(pw_corr[i,:][pw_corr[i,:]>0])
                    maxi=0
                    for j in range(len(seeds)):
                        if(pw_corr[np.where(idx2==seeds[j][0])[0],i] > maxd):
                            maxd=pw_corr[np.where(idx2==seeds[j][0])[0],i]
                            maxi=j
                            flag=True
                    if(not flag):
                        seeds.append(np.array([idx2[i]]))
                    else:
                        seeds[maxi]=np.append(seeds[maxi],idx2[i])

                self.gene_groups=seeds
                self.geneID_groups=[]
                for i in range(len(self.gene_groups)):
                    self.geneID_groups.append(self.gene_names[self.gene_groups[i]])
                    
                return seeds

    
    
    def run_tsne(self,metric='precomputed',**kwargs):
        """Wrapper for sklearn's t-SNE implementation.
        
        See sklearn for the t-SNE documentation. All arguments are the same
        with the exception that 'metric' is set to 'precomputed' by default, 
        implying that this function expects a distance matrix by default.        
        """
        dt=man.TSNE(metric=metric,**kwargs).fit_transform(self.dist)
        self.dt=dt
            
    def scatter(self,projection = None,c=None,cmap='rainbow',new_figure=True,colorbar=True,**kwargs):
        """Display a scatter plot.
        
        Displays a scatter plot using the SAM projection or another input
        projection with or without annotations.
        
        Parameters
        ----------
        
        projection - ndarray of floats, optional, default None
            An N x 2 matrix, where N is the number of data points. If not 
            specified, use the SAM projection instead. Otherwise, display the 
            input projection.
        
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
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after loading the data.")
        else:
            if(projection is None):
                if(self.dt is None):
                    print("Please create a 2D projection first using 'run_tsne'. ")            
                    return ;
                else:
                    dt=self.dt
            else:
                dt=projection

            if(new_figure):
                plt.figure();
            if(c is None):
                plt.scatter(dt[:,0],dt[:,1],**kwargs)
            else:            
                if(type(c[0]) is str or type(c[0]) is np.str_):
                    i = ut.convert_annotations(c)    
                    ui,ai=np.unique(i,return_index=True)
                    cax=plt.scatter(dt[:,0],dt[:,1],c=i,cmap=cmap,**kwargs)        

                    if(colorbar):
                        cbar = plt.colorbar(cax, ticks=ui)
                        cbar.ax.set_yticklabels(c[ai])
                else:
                    i = c
                    plt.scatter(dt[:,0],dt[:,1],c=i,cmap=cmap,**kwargs)                    

                    if(colorbar):
                        plt.colorbar();                       



    def show_gene_expression(self,gene,projection = None,new_figure=True,average_exp=True,colorbar=True,**kwargs):
        """Display a gene's expressions.
        
        Displays a scatter plot using the SAM projection or another input
        projection with a particular gene's expressions overlaid.
        
        Parameters
        ----------
        
        projection - ndarray of floats, optional, default None
            An N x 2 matrix, where N is the number of data points. If not 
            specified, use the SAM projection instead. Otherwise, display the 
            input projection.
        
        new_figure - bool, optional, default True
            If True, creates a new figure. Otherwise, outputs the scatter plot
            to currently active matplotlib axes.
        
        average_exp - bool, optional, default True
            If True, the plots use the k-nearest-neighbor-averaged expression 
            values to smooth out noisy expression patterns and improves
            visualization. 
        
        colorbar - bool, optional default True
            If True, display a colorbar indicating which values / annotations
            correspond to which color in the scatter plot.
            
        Keyword arguments - 
            All other keyword arguments that can be passed into 
            matplotlib.pyplot.scatter can be used.
        """
        
        if (not self.analysis_performed):
            print("Please run the SAM analysis first using 'run' after loading the data.")
        else:
            if(projection is None):
                if(self.dt is None):
                    print("Please create a 2D projection first using 'run_tsne'. ")            
                    return ;
                else:
                    dt=self.dt
            else:
                dt = projection


            if(type(gene)==str or type(gene)==np.str_):
                idx=np.where(self.gene_names==gene)[0]
                name=gene
                if(idx.size==0):
                    print("Gene note found in the filtered dataset. Note that genes are case sensitive.")
                    return
            else:
                idx=gene
                name=self.gene_names[idx]

            if(new_figure):
                plt.figure(); 
            if(average_exp):
                a=self.D_avg[:,idx].flatten()
            else:
                a=self.D[:,idx].flatten()
            
            plt.scatter(dt[:,0],dt[:,1],c=a,cmap='viridis',**kwargs)
            plt.title(name)
            if(colorbar):
                plt.colorbar();

