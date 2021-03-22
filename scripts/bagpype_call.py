import os
import sys
import pandas as pd
import sklearn
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.model_selection import train_test_split
from time import time
from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from igraph import *
import subprocess
from sklearn import datasets
from sklearn.metrics import *
import shutil
import yaml
params = yaml.safe_load(open('bagpype_template.yaml'))

def match(C):
    """
    Replication of R's match function
    Returns a vector of the positions of (first) matches of its first argument in its second
    """
    
    u, ui = np.unique(C, return_index=True)
    ui = sorted(ui)
    newC = np.zeros(C.shape)
    for i in range(len(ui)):
        newC[np.where(C == C[ui[i]])[0]] = i + 1
        
    return newC


def outer_equal(x):
    """
    Replication of R's outer function with FUN param '=='
    """
    
    res = np.zeros((x.shape[0], x.shape[0]))
    for i in range(x.shape[0]):
        res[i,:] = x == x[i]
    
    return res


def hadamard(nearest):
    """
    Transform data into Hadamard distance matrix
    """
    
    common = nearest.T @ nearest
    ranks = np.outer(np.diag(common), np.ones(nearest.shape[0]))
    neighborUnion = ranks + ranks.T - common
    G = common / neighborUnion
    np.fill_diagonal(G, 0)
    
    return G


def topMax(x, N):
    """
    find Nth largest number in an array
    """
    
    L = len(x)
    assert N < L, 'Number of neighbors cannot be larger than length of data'
    
    while L != 1:
        initial_guess = x[0]
        top_list = x[x > initial_guess]
        bottom_list = x[x < initial_guess]
        
        topL = len(top_list)
        bottomL = len(bottom_list)
        
        if (topL < N) and (L - bottomL >= N):
            x = initial_guess
            break
        
        if topL >= N:
            x = top_list
        else:
            x = bottom_list
            N = N - L + bottomL
        L = len(x)
    
    return x


def bottomMin(x, N):
    """
    find Nth smallest number in an array
    """
    
    return np.round(topMax(x, len(x) - N + 1))


def find_neighbors(D, k):
    """
    Tranform distance matrix to binary k-nearest neighbors graph
    """
    
    nearest = np.zeros(D.shape)
    for i in range(nearest.shape[1]):
        nearest[:,i] = np.round(D[:,i]) <= bottomMin(np.round(D[:,i]), k)
    
    return nearest


def modularity(G, C):
    """
    Calculate graph's modularity
    """
    
    m = np.sum(G)
    Ki = np.repeat(np.sum(G, axis=1), G.shape[1]).reshape(G.shape)
    Kj = Ki.T
    delta_function = outer_equal(C)
    Q = (G - Ki * Kj / m) * delta_function / m
    return np.sum(Q)


def delta_modularity(G, C, i, m, Kj, newC=None):
    """
    Calculate change in modularity by adding a node to a cluster or by removing it
    """
    
    c = np.copy(C)
    
    if newC is None: # removing a node from C
        # removing a solitary node from a cluster does not change the modularity
        if np.sum(c == c[i]) == 1:
            return 0
        newC = c[i]
        c[i] = np.max(c) + 1
        
    I = c == newC
    Ki = np.sum(G[i,:]) # the sum of all the edges connected to node i
    Ki_in = np.sum(G[i,I]) # the sum of all the edges conneting node i to C
    Kjs = np.sum(Kj[I])
    deltaQ = Ki_in / m - Ki * Kjs / m**2
    
    return deltaQ * 2


def louvain_step(G, C, O, Q=None):
    """
    Run a single step of the Louvain algorithm
    """
    
    if not Q:
        Q = modularity(G, C)
        
    m = np.sum(G)
    Kj = np.sum(G, axis=1)
    
    for i in O:
        reassign = np.array(list(set(C[G[i,:] > 0]).difference(set([C[i]]))))
        
        if len(reassign) != 0:
            delta_remove = delta_modularity(G, C, i, m, Kj)

            deltaQ = np.array([delta_modularity(G, C, i, m, Kj, newC=x) for x in reassign])

            if np.max(deltaQ) - delta_remove > 0:
                idx = np.argmax(deltaQ)
                Q += np.max(deltaQ) - delta_remove
                if reassign[idx] > G.shape[0]:
                    print(i)
                C[i] = reassign[idx]
            if delta_remove < 0:
                C[i] = np.max(C) + 1
                Q -= delta_remove
    
    return C, Q


def louvain(G, C, maxreps=100):
    """
    Run Louvain algorithm
    """
    
    assert np.allclose(G, G.T), 'Input graph must be symmetric'
    
    order = np.random.permutation(G.shape[1])
    Q = modularity(G, C)
    
    for i in range(maxreps):
        C, Q = louvain_step(G, C, order, Q)
        C = match(C)
        
        # run the second phase, trying to combine each cluster
        n_clust = len(np.unique(C))
        metaG = np.zeros((n_clust, n_clust))
        for ci in range(1, n_clust + 1):
            G_tmp = G[C == ci]
            for cj in range(ci, n_clust + 1):
                G_tmp2 = G_tmp[:,C == cj]
                metaG[ci-1,cj-1] = np.sum(G_tmp2)
                metaG[cj-1,ci-1] = metaG[ci-1,cj-1]
                
        metaC, metaQ = louvain_step(metaG, np.arange(n_clust) + 1, np.random.permutation(n_clust))
        if metaQ - Q > 1e-15:
            tempC = np.copy(C)
            for ci in range(1, n_clust + 1):
                tempC[C == ci] = metaC[ci-1]
            C = match(tempC)
            Q = metaQ
        else:
            break
            
    return C, Q
        
def pheno_clust(filepath=None, subset=None, X=None, outfile=None, repeats=50, verbose=True, distance=None):
    """
    Run entire phenoClust pipeline:                                                                     
    This method clusters sample data using a two-step approach. In the first step it converts the high dimensional data into a knn-nearest neighbor graph, 
    thus connecting samples that have a similar profile. The edges in knn-nearest neighbors graph are given weights according also known as the Hadamard
    index. In the second step the Louvaine algorithm, which was developed to find communities in social networks, is applied to this graph in order to find
    samples that share similar spaces in multi-dimensional space.
    """
    assert filepath or X is not None, 'Either a matrix or a filepath to the data must be passed in'
    
    if verbose:
        tic = time()
    
    if filepath:
        assert subset, 'If filepath to dataframe is passed, you must also include the subset of columns that you intend to pass into the algorithm'
        
        df = pd.read_csv(filepath)
        df = df[subset]
        df.dropna(inplace=True)

        X = np.array(df)
        
    X_Z = (X - np.mean(X, axis=0, keepdims=True)) / np.std(X, axis=0, dtype=np.float64, ddof=1, keepdims=True)
    
    if distance == "spearman":  
        D, rho = spearmanr(X_Z, axis=1)
        
    if distance == "euclid":
        D = euclidean_distances(X_Z)
        
    D = np.round((X_Z.shape[1]**3 - X_Z.shape[1]) * (1 - D) / 6).astype(np.int)
    
    b = np.ceil(np.log2(X_Z.shape[0]) + 1)
    k = np.ceil(X_Z.shape[0] / b).astype(np.int)
    nearest = find_neighbors(D, k + 1)
    G = hadamard(nearest)
    C = np.arange(G.shape[1]) + 1
    
    Q = -np.inf
    for t in range(repeats):
        C, newQ = louvain(G, C)
        if newQ > Q:
            Q = newQ
            if verbose:
                print('iter: %d/%d; Q = %.5f; # of communities: %d' % (t+1, repeats, Q, len(np.unique(C))))
        else:
            break
    
    if verbose:
        toc = time()
        print('BagPype completed in: %.2f seconds' % (toc - tic))
    
    return C, Q


def bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir, out_dir_stab, subset, silo_metric):
    '''csv_folder: folder that contains all the csv output from python louvain.
       split: string in the csv files' names that specifies the split eg.Split_1 '''
    
    my_data = [pd.read_csv(csv_folder+file,index_col=0) for file in os.listdir(csv_folder) if split in file] # list contains all the csv in dataframe format
    for i in range(len(my_data)): # loop over my_data
        my_data[i] = my_data[i].rename(columns = {"0":'Key', "1":'cluster'})
        boot=my_data[i].drop_duplicates() #remove duplicates
        df_merge = boot.merge(boot, on='cluster') # create adjacency matrix step1 
        this_adj =pd.crosstab(df_merge.Key_x, df_merge.Key_y) # create adjacency matrix step2
        np.fill_diagonal(this_adj.values, 0) #set the diagonal of the adjacency matrix to 0
        if i ==0:
            adj_full=this_adj #if this is the first adjacency matrix set it as the initial adj_full 
        else:
            adj_full=adj_full.add(this_adj,fill_value=0) # add this adjacency matrix to the full adjacency matrix
        boot_mask=my_data[i].drop_duplicates()
        boot_mask.cluster=boot_mask.cluster.replace(boot_mask.cluster.unique(),np.ones(boot_mask.cluster.unique().shape))#replace all clusters with 1
        df_merge2 = boot_mask.merge(boot_mask, on='cluster')
        mask_adj =pd.crosstab(df_merge2.Key_x, df_merge2.Key_y)
        np.fill_diagonal(mask_adj.values, 0) #set the diagonal of the adjacency matrix to 0
        if i ==0:
            mask_full= mask_adj  #if this is the first adjacency matrix set it as the initial mask_full 
        else:
            mask_full=mask_full.add( mask_adj,fill_value=0) # add this mask adjacency matrix to the full mask adjacency matrix
    stab_full = adj_full.div(mask_full)
    np.fill_diagonal(stab_full.values, 0)
    #Create df Id from matrix column names
    columnsNamesArr = pd.DataFrame(stab_full.columns.values)
    # Final Louvain Clustering Split 1 
    graph = Graph.Weighted_Adjacency(stab_full.values.tolist(), mode=ADJ_UNDIRECTED, attr="weight")
    Louvain = graph.community_multilevel(weights=graph.es['weight'])
    #Louvain.membership 
    Q = graph.modularity(Louvain, weights=graph.es['weight']); print(Q)
    # Create dataframe of Subtypes for Split 1 
    Subtypes = pd.DataFrame(Louvain.membership); Subtypes = Subtypes + 1; Subtypes.loc[:, 'Q'] = Q
    # create dataframe of Split 1 Subtypes 
    subs = pd.concat([columnsNamesArr.reset_index(drop=True), Subtypes], axis=1); subs.columns = ['Key', 'Subtype', 'Q']
    # read in Split 1 dataframe 
    df = pd.read_csv(data_frame_path); df = df.rename(columns={'Unnamed: 0': 'Key'})
    # create Split 1 
    Split_1 = pd.merge(subs, df, on='Key'); Split_1.to_csv(out_dir); stab_full.to_csv(out_dir_stab)
    return stab_full, Split_1, Q


def bagpype(outfolder, batch, path, n_straps, ID, Boot=None, louvain_metric=None, silo_metric=None):
    #______________________________________________________________________________________________________________
    # set up 
    batch = batch; df1 = pd.read_csv(path); silo_metric=silo_metric
    #df1['Key'] = list(range(len(df1))) 
    if Boot == 'No': 
        os.system(f'mkdir {outfolder}/Output/'); os.system(f'mkdir {outfolder}/Output/Results/'); os.system(f'mkdir {outfolder}/Output/Results/NonBoot')
        #___________________________________________________________________________________________________________________________________
        df1 = df1.drop(['Unnamed: 0'], axis=1); subset = df1.columns[df1.columns != ID]; y = np.array(df1[ID],dtype='str') 
        communities, Q = pheno_clust(X=np.array(df1[subset]).astype(np.float64), verbose=False, distance = louvain_metric)
        score = silhouette_score(np.array(df1[subset]), communities, metric = silo_metric) 
        df2 = {ID:y,'Subtype':communities}; df2 =pd.DataFrame(df2); df2.loc[:, 'Q'] = Q; df2.loc[:, 'Silo'] = score
        df2 = pd.merge(df2, df1, on = ID); df2.to_csv(f'{outfolder}/Output/Results/NonBoot/NonBoot_{batch}_Full_Subtypes.csv'); print(Q)
        #___________________________________________________________________________________________________________________________________
    
    if Boot == 'Both':
        df1 = df1.drop([ID], axis=1); df1 = df1.rename(columns={'Unnamed: 0': 'Key'}); subset = df1.columns[df1.columns != 'Key']
        df = df1[subset]; y = np.array(df1['Key'],dtype='str'); n_straps = n_straps; n = df.shape[0]; b_idx = np.zeros((n_straps, n))
        #___________________________________________________________________________________________________________________________________
        # bootstrapping
        for i in range(n_straps):
            random_state = np.random.RandomState(); b_idx[i] = random_state.randint(0, high= n - 1, size=n)
        b_idx = b_idx.astype(np.int); y_boot = np.zeros(b_idx.shape)
        for i in range(b_idx.shape[0]):
            y_boot[i] = y[b_idx[i]]
        #___________________________________________________________________________________________________________________________________
        #create directories
        os.system(f'mkdir {outfolder}/Data/'); os.system(f'mkdir {outfolder}/Output/'); os.system(f'mkdir {outfolder}/Output/Results/')
        os.system(f'mkdir {outfolder}/Output/Results/Boot'); os.system(f'mkdir {outfolder}/Output/Results/Boot_Q'); os.system(f'mkdir {outfolder}/Output/Results/Boot_Silo')  
        #save data 
        df.to_csv(f'{outfolder}/Data/temp.csv'); np.save(f'{outfolder}/Output/Results/y_boot.npy',y_boot); np.save(f'{outfolder}/Output/Results/b_idx.npy',b_idx)
        temp_txtfile = np.arange(0,n_straps,1).astype(str); np.savetxt(f'{outfolder}/Output/Results/temp_txtfile.txt', temp_txtfile,fmt='%s')
        #___________________________________________________________________________________________________________________________________
        #parallel call 
        tstart = time()
        pcall = 'bash ' + params['pype_path'] + '/BagPype/scripts/bagpype_bash_parallel.sh %s' % batch
        subprocess.check_call(pcall, shell=True); tend = time(); print('Time to run %s bootstraps: ' % n_straps, tend-tstart)
        #___________________________________________________________________________________________________________________________________
        #Bagging 
        csv_folder = f'{outfolder}/Output/Results/Boot/'; split = batch; data_frame_path = path
        out_dir = f'{outfolder}/Output/Results/{batch}_Full_Subtypes.csv'; out_dir_stab = f'{outfolder}/Output/Results/{batch}_Full_Stab.csv'
        stab_full, Split_1, Q = bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir, out_dir_stab, subset, silo_metric)
        #___________________________________________________________________________________________________________________________________  
        os.system(f'mkdir {outfolder}/Output/Results/NonBoot')
        #___________________________________________________________________________________________________________________________________   
        df1 = pd.read_csv(path); df1 = df1.drop(['Unnamed: 0'], axis=1); subset = df1.columns[df1.columns != ID]; y = np.array(df1[ID],dtype='str') 
        communities, Q = pheno_clust(X=np.array(df1[subset]).astype(np.float64), verbose=False, distance = louvain_metric)
        score = silhouette_score(np.array(df1[subset]), communities, metric = silo_metric) 
        df2 = {ID:y,'Subtype':communities}; df2 =pd.DataFrame(df2); df2.loc[:, 'Q'] = Q; df2.loc[:, 'Silo'] = score
        df2 = pd.merge(df2, df1, on = ID); df2.to_csv(f'{outfolder}/Output/Results/NonBoot/NonBoot_{batch}_Full_Subtypes.csv'); print(Q)       
        
    if Boot == 'Yes':
        df1 = df1.drop([ID], axis=1); df1 = df1.rename(columns={'Unnamed: 0': 'Key'}); subset = df1.columns[df1.columns != 'Key']
        df = df1[subset]; y = np.array(df1['Key'],dtype='str'); n_straps = n_straps; n = df.shape[0]; b_idx = np.zeros((n_straps, n))
        #___________________________________________________________________________________________________________________________________
        # bootstrapping
        for i in range(n_straps):
            random_state = np.random.RandomState(); b_idx[i] = random_state.randint(0, high= n - 1, size=n)
        b_idx = b_idx.astype(np.int); y_boot = np.zeros(b_idx.shape)
        for i in range(b_idx.shape[0]):
            y_boot[i] = y[b_idx[i]]
        #___________________________________________________________________________________________________________________________________
        #create directories
        os.system(f'mkdir {outfolder}/Data/'); os.system(f'mkdir {outfolder}/Output/'); os.system(f'mkdir {outfolder}/Output/Results/')
        os.system(f'mkdir {outfolder}/Output/Results/Boot'); os.system(f'mkdir {outfolder}/Output/Results/Boot_Q'); os.system(f'mkdir {outfolder}/Output/Results/Boot_Silo')  
        #save data 
        df.to_csv(f'{outfolder}/Data/temp.csv'); np.save(f'{outfolder}/Output/Results/y_boot.npy',y_boot); np.save(f'{outfolder}/Output/Results/b_idx.npy',b_idx)
        temp_txtfile = np.arange(0,n_straps,1).astype(str); np.savetxt(f'{outfolder}/Output/Results/temp_txtfile.txt', temp_txtfile,fmt='%s')
        #___________________________________________________________________________________________________________________________________
        #parallel call 
        tstart = time()
        pcall = 'bash ' + params['pype_path'] + '/BagPype/scripts/bagpype_bash_parallel.sh %s' % batch
        subprocess.check_call(pcall, shell=True); tend = time(); print('Time to run %s bootstraps: ' % n_straps, tend-tstart)
        #___________________________________________________________________________________________________________________________________
        #Bagging 
        csv_folder = f'{outfolder}/Output/Results/Boot/'; split = batch; data_frame_path = path
        out_dir = f'{outfolder}/Output/Results/{batch}_Full_Subtypes.csv'; out_dir_stab = f'{outfolder}/Output/Results/{batch}_Full_Stab.csv'
        stab_full, Split_1, Q = bagging_adjacency_matrixes(csv_folder, split, data_frame_path, out_dir, out_dir_stab, subset, silo_metric)
        #___________________________________________________________________________________________________________________________________

