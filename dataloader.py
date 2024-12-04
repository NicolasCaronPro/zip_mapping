import random
from model import *
from torch_geometric.data import Dataset
from torch.utils.data import DataLoader

def read_stations_zones(path):
    gpd1 = gpd.read_file(path / 'viginond_stations.geojson')
    #gpd2 = gpd.read_file(path / 'viginond_zip_modified.geojson')
    gpd2 = gpd.read_file(path / 'viginond_zip.geojson')
    return gpd1, gpd2

def split_zones(stations, zones, ratio_test = 0.2, ratio_val=0.2):
    departements_list = list(stations.departement.unique())
    random.seed(42)

    # Calculer 10 % de la liste
    sample_size = int(len(departements_list) * ratio_test)

    # Sélectionner aléatoirement 10 % des départements
    train_departements = random.sample(departements_list, sample_size)
    val_departements = random.sample(train_departements * ratio_val)
    test_departement = [dept for dept in departements_list if dept not in train_departements and dept not in val_departements]

    logger.info(f' Train departement selected : {train_departements}')
    logger.info(f' Val departement selected : {val_departements}')
    logger.info(f' Test departement selected : {val_departements}')

    stations['set'] = None
    stations.loc[stations[stations['departement'].isin(train_departements)].index, 'set'] = 'train'
    stations.loc[stations[stations['departement'].isin(val_departements)].index, 'set'] = 'val'
    stations.loc[stations[stations['departement'].isin(test_departement)].index, 'set'] = 'test'

    zones = zones.set_index('station').join(stations.set_index('station')['set'], on='station').reset_index()

    train_mask = zones[zones['set'] == 'train']
    val_mask = zones[zones['set'] == 'val']
    test_mask = zones[zones['set'] == 'test']

    return zones[train_mask].reset_index(drop=True), zones[val_mask].reset_index(drop=True), zones[test_mask].reset_index(drop=True)

#################################################### Torch loader ####################################################

class ReadDataset(Dataset):
    def __init__(self, X : list,
                 Y : list,
                 test : bool,
                 leni : int,
                 device : torch.device,
                 path : Path) -> None:
        
        self.X = X
        self.Y = Y
        self.device = device
        self.leni = leni
        self.path = path
        self.test = test

    def __getitem__(self, index) -> tuple:
        x = read_object(self.X[index], self.path)
        y = read_object(self.Y[index], self.path)

        if self.test:
            return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device), self.X[index].split('.')[0]

        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device), \
            torch.tensor(y, dtype=torch.float32, device=self.device),

    def __len__(self) -> int:
        return self.leni
    
    def len(self):
        pass

    def get(self):
        pass

def create_loader(zones, device, dir_output, test):
    dir_database = dir_output / 'database' 
    stations = zones['stations'].unique()
    X = []
    y = []
    for station in stations:
        X.append(f'{station}_features.pkl')
        y.append(f'{station}.pkl')
    
    return ReadDataset(X, y, test, len(X), device, dir_database)

def create_train_val_test_loader(train_zone, val_zone, test_zone, device, dir_output):
    
    train_loader = create_loader(train_zone, device, dir_output, False)
    val_loader = create_loader(val_zone, device, dir_output, False)
    test_loader = create_loader(test_zone, device, dir_output, True)

    return train_loader, val_loader, test_loader