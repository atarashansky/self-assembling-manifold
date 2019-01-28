import numpy as np
import scipy as sp
import os, errno
from sklearn.decomposition import PCA,TruncatedSVD
import umap.distances as dist

import umap.sparse as sparse
from umap.rp_tree import rptree_leaf_array, make_forest
from umap.nndescent import (
    make_nn_descent,
)

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1

__version__ = '0.3.3'




def nearest_neighbors(X,n_neighbors=15,seed=0,metric='correlation'):
    
    distance_func = dist.named_distances[metric]


    if metric in ("cosine", "correlation", "dice", "jaccard"):
        angular = True
    else:
        angular=False

    
    random_state = np.random.RandomState(seed=seed)
    rng_state = random_state.randint(INT32_MIN, INT32_MAX, 3).astype(np.int64)
    
    metric_nn_descent = make_nn_descent(
        distance_func, tuple({}.values())
    )
    
    n_trees = 5 + int(round((X.shape[0]) ** 0.5 / 20.0))
    n_iters = max(5, int(round(np.log2(X.shape[0]))))

    rp_forest = make_forest(X, n_neighbors, n_trees, rng_state, angular)
    leaf_array = rptree_leaf_array(rp_forest)
    knn_indices, knn_dists = metric_nn_descent(
        X,
        n_neighbors,
        rng_state,
        max_candidates=60,
        rp_tree_init=True,
        leaf_array=leaf_array,
        n_iters=n_iters,
        verbose=False,
    )
    return knn_indices,knn_dists


def knndist(nnma):
    knn=[]
    for i in range(nnma.shape[0]):
        knn.append(np.where(nnma[i,:]==1)[0])
    knn=np.vstack(knn)
    dist=np.ones(knn.shape)
    return knn,dist

def weighted_PCA(mat,do_weight=True,npcs=None,solver='auto'):
    mat = (mat - np.mean(mat,axis=0))
    if(do_weight):
        if(min(mat.shape)>=10000 and npcs is None):
            print("More than 10,000 cells. Running with 'npcs' set to < 1000 is recommended.")
            
        if(npcs is None):
            ncom=min(mat.shape)
        else:
            ncom=min((min(mat.shape),npcs))
        
        pca = PCA(svd_solver=solver,n_components=ncom)        
        reduced=pca.fit_transform(mat)
        scaled_eigenvalues=reduced.var(0)#pca.explained_variance_/pca.explained_variance_.max()
        scaled_eigenvalues=scaled_eigenvalues/scaled_eigenvalues.max()
        reduced_weighted=reduced*scaled_eigenvalues[None,:]**0.5
    else:
        pca = PCA(n_components=npcs,svd_solver=solver)
        reduced=pca.fit_transform(mat)
        if reduced.shape[1]==1:
            pca = PCA(n_components=2,svd_solver=solver)
            reduced=pca.fit_transform(mat)
        reduced_weighted=reduced
        
    return reduced_weighted,pca

def weighted_sparse_PCA(mat,do_weight=True,npcs=None):

    if(do_weight):
        if(min(mat.shape)>=10000 and npcs is None):
            print("More than 10,000 cells. Running with 'npcs' set to < 1000 is recommended.")
            
        if(npcs is None):
            ncom=min(mat.shape)
        else:
            ncom=min((min(mat.shape),npcs))
        
        pca = TruncatedSVD(n_components=ncom)        
        reduced=pca.fit_transform(mat)
        scaled_eigenvalues=reduced.var(0)#pca.explained_variance_/pca.explained_variance_.max()
        scaled_eigenvalues=scaled_eigenvalues/scaled_eigenvalues.max()
        reduced_weighted=reduced*scaled_eigenvalues[None,:]**0.5
    else:
        pca = TruncatedSVD(n_components=npcs,svd_solver='auto')
        reduced=pca.fit_transform(mat)
        if reduced.shape[1]==1:
            pca = TruncatedSVD(n_components=2,svd_solver='auto')
            reduced=pca.fit_transform(mat)
        reduced_weighted=reduced
        
    return reduced_weighted,pca

def transform_wPCA(mat,pca):
    mat = (mat - mat.mean(0))
    reduced=mat.dot(pca.components_.T)
    """
    scale by new variance as opposed to old?
    """
    v = pca.explained_variance_
    #v = reduced.var(0)
    scaled_eigenvalues=v/v.max()
    reduced_weighted=np.array(reduced)*scaled_eigenvalues[None,:]**0.5
    return reduced_weighted
         
def search_string(vec,s):
    m=[]
    s=s.lower()
    for i in range(len(vec)):
        st=vec[i].lower()
        b=st.find(s)
        if(b!=-1):
            m.append(i)
    if(len(m)>0):
        return vec[np.array(m)],np.array(m)
    else:
        return [-1,-1]

def distance_matrix_error(dist1,dist2):
    s=0
    for k in range(dist1.shape[0]):
        s+=np.corrcoef(dist1[k,:],dist2[k,:])[0,1]
    return 1-s / dist1.shape[0]


def extract_annotation(cn,x,c='_'):
    m=[]
    for i in range(cn.size):
        m.append(cn[i].split(c)[x])
    return np.array(m)

def isolate(dt,x1,x2,y1,y2):
    return np.where(np.logical_and(np.logical_and(dt[:,0]>x1,dt[:,0]<x2),np.logical_and(dt[:,1]>y1,dt[:,1]<y2)))[0]

def to_lower(y):
    x=y.copy().flatten()
    for i in range(x.size):    
        x[i]=x[i].lower()
    return x
def to_upper(y):
    x=y.copy().flatten()
    for i in range(x.size):    
        x[i]=x[i].upper()
    return x    
def create_folder(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
            

def convert_annotations(A):
    x=np.unique(A)
    y=np.zeros(A.size)
    z=0
    for i in x:
        y[A==i]=z
        z+=1    
    
    return y.astype('int')

 
   

def compute_distances(A,dm):
    if(dm=='euclidean'):
        m=np.dot(A,A.T)
        h=np.diag(m)
        x=h[:,None]+h[None,:] - 2*m
        x[x<0]=0
        dist=np.sqrt(x)
    elif(dm=='correlation'):
        dist=1-np.corrcoef(A)
    else:
        dist = sp.spatial.distance.squareform(sp.spatial.distance.pdist(A,
                                                    metric=dm))
    return dist

def dist_to_nn(d,K):
    E=d.copy()
    np.fill_diagonal(E,-1)
    M=np.max(E)*2    
    x=np.argsort(E,axis=1)[:,:K]
    E[np.tile(np.arange(E.shape[0]).reshape(E.shape[0],-1),
              (1,x.shape[1])).flatten(),x.flatten()]=M
        
    E[E<M]=0
    E[E>0]=1
    return E#,x


