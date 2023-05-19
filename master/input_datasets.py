from torch_geometric.datasets import TUDataset

def input_d(name:str):
    if name in ['MUTAG', 'IMDB-BINARY', 'NCI1', 'PROTEINS', 'REDDIT-BINARY']:
        return  TUDataset(root=f'/tmp/{name}', name=f'{name}')
    if name in ['PTC_FM', 'PTC_MR']:
        return  TUDataset(root=f'/tmp/{name}', name=f'{name}')[1:]
    else:
        return "Dataset not found"