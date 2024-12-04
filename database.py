from dataloader import *

def create_zones(stations, zones, resolution, dir_output):
    lats = resolutions[resolution]['y']
    longs = resolutions[resolution]['x']
    check_and_create_path(dir_output / 'database')
    ustations = stations.station.unique()
    for station in ustations:
        station_gdf = zones[zones['station'] == station]
        
        data = rasterisation(station_gdf, lats, longs, column='cluster', defval = 0, name='default', dir_output='/media/caron/X9 Pro/corbeille')
        os.remove(Path('/media/caron/X9 Pro/corbeille') / 'default.tif')
        save_object(data, f'{station}.pkl', dir_output / 'zones')

def create_features(stations, zones, resolution, features, dir_output):
    lats = resolutions[resolution]['y']
    longs = resolutions[resolution]['x']
    check_and_create_path(dir_output / 'database')
    ustations = stations.station.unique()
    
    for station in ustations:
        departement = stations[stations['station'] == station]['departement'].values[0]
        root_data = root_features / departement / 'data'
        
        station_zones = zones[zones['station'] == station]
        bbox_zone = station_zones.total_bounds  # Calcul de la bounding box [minx, miny, maxx, maxy]

        for feature in features:
            feature_data = gpd.read_file(root_data / f'{feature}.geojson')
            
            # Filtrer uniquement les éléments du feature_data qui intersectent la bounding box pour optimiser
            feature_data = feature_data.cx[bbox_zone[0]:bbox_zone[2], bbox_zone[1]:bbox_zone[3]]

            if 'data' not in locals():
                # Rasteriser et enregistrer les données
                data = rasterisation(feature_data, lats, longs, column='cluster', defval=0, name='default', dir_output='/media/caron/X9 Pro/corbeille')
            else:
                data_ = rasterisation(feature_data, lats, longs, column='cluster', defval=0, name='default', dir_output='/media/caron/X9 Pro/corbeille')
                data = np.concatenate((data, data_), axis=0)

            # Supprimer le fichier temporaire .tif
            os.remove(Path('/media/caron/X9 Pro/corbeille') / 'default.tif')

        # Sauvegarder l'objet en pickle
        save_object(data, f'{station}.pkl', dir_output / 'zones')

def create_database(stations, zones, dir_output, resolution, features, doFeatures, doZones):
    dir_output = dir_output / 'database'
    check_and_create_path(dir_output)

    ############################## Zones ##############################

    if doZones:
        create_zones(stations, zones, resolution, dir_output)

    ############################## Features ##############################

    if doFeatures:
        create_features(stations, zones, resolution, features, dir_output)