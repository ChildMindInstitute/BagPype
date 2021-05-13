from bagpype_call import *

i = sys.argv[1]; i = int(i) # Load boostrap number
batch = sys.argv[2]; batch = np.str(batch) # load batch name
batch_name = f'{batch}'
batch_path = params['data_path']
outfolder= batch_path+'/'+batch_name 
n_straps = params['n_straps']
louvain_metric = params['louvain_metric']
silo_metric = params['silo_metric']
k = params['k']
            
df = pd.read_csv(f'{outfolder}/Data/temp.csv')
df = df.rename(columns={'Unnamed: 0': 'Key'})
subset = df.columns[df.columns != 'Key']
y_boot = np.load(f'{outfolder}/Output/Results/y_boot.npy') # load indentifer key 
b_idx = np.load(f'{outfolder}/Output/Results/b_idx.npy') # load bootstrap index number
X_split = df.iloc[b_idx[i],:]

#Run louvain
print('Lacing up bootstrap #%d' % (i + 1))
communities2, Q2 = pheno_clust(X=np.array(X_split[subset]), verbose=True, distance = louvain_metric, k=k)
# silouette score
score = silhouette_score(np.array(X_split[subset]), communities2, metric = silo_metric) 

data1 = np.hstack((y_boot[i].reshape(-1,1), communities2.reshape(-1,1)))
res1 = pd.DataFrame(data=data1)
res1.to_csv(f'{outfolder}/Output/Results/Boot/%s_%s.csv' % (batch, i))

for x in range(n_straps):
    for t in range(2):
        Q = np.array([Q2] * t)
        Silo = np.array([score] * t)
    
data2 = np.hstack(Q.reshape(-1,1)); res2 = pd.DataFrame(data=data2)
res2.to_csv(f'{outfolder}/Output/Results/Boot_Q/%s_%s.csv' % (batch, i))
             
data3 = np.hstack(Silo.reshape(-1,1)); res3 = pd.DataFrame(data=data3)
res3.to_csv(f'{outfolder}/Output/Results/Boot_Silo/%s_%s.csv' % (batch, i))

#______________________________________________________________________________________________________________________________________________________
### Beta testing for Variable Clusering ### 
#X_split = df.iloc[b_idx[i],:]
#X_split = X_split.T
#X_split = X_split.iloc[1:]
#print('Lacing up right shoe bootstrap #%d' % (i + 1))
#communities2, Q2 = pheno_clust(X=np.array(X_split), verbose=True, distance = louvain_metric)
#data1 = np.hstack(communities2.reshape(-1,1))
#res1 = pd.DataFrame(data=data1)
#res1 = res1.rename(columns={0: '1'})
#res1[0] = list(range(len(res1)))
#res1 = res1.iloc[:,[1,0]]
#res1.to_csv(f'{outfolder}/Output/data/Boot/%s_%s.csv' % (batch, i))