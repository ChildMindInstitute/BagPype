from bagpype_call import *
outfolder = (); path = ()
for i in params['batch']:
    batch_path = params['data_path']
    batch_name = f'{i}'
    outfolder= batch_path+'/'+batch_name 
    path = f'{outfolder}/{i}.csv' 
    os.system(f'mkdir -p {batch_path}/{i}')
    for d in [batch_path+"/"+batch_name +'.csv']:
        shutil.copy(d, outfolder)
    bagpype(outfolder = outfolder,
            batch = i, 
            path = path, 
            n_straps = params['n_straps'], 
            ID = params['ID'], 
            Boot = params['Boot'], 
            algorithm = params['algorithm'], 
            distance_metric = params['distance_metric'], 
            silo_metric = params['silo_metric'],
            k = params['k'],
            resolution_parameter=params['resolution_parameter']) 
    
    
