from sympy import EX
from visualiize import *
from shapely.geometry import Point, Polygon
from shapely.geometry import box

def add_new_zones_from_points(zones, new_points, size, stations, geo_france):
    """
    Ajoute de nouvelles zones carrées dans un GeoDataFrame existant et met à jour le GeoDataFrame des stations.

    Parameters:
    - zones (gpd.GeoDataFrame): Le GeoDataFrame existant contenant les zones.
    - new_points (list of tuple): Liste de points (x, y) représentant les centres des nouvelles zones.
    - size (float): Taille du côté des carrés à ajouter.
    - stations (gpd.GeoDataFrame): GeoDataFrame des stations à mettre à jour.
    - geo_france (gpd.GeoDataFrame): GeoDataFrame contenant les départements de France.

    Returns:
    - zones_updated (gpd.GeoDataFrame): Le GeoDataFrame mis à jour avec les nouvelles zones.
    - stations (gpd.GeoDataFrame): Le GeoDataFrame des stations mis à jour avec les nouvelles stations.
    """

    half_size = size / 2

    new_zones_geom = []
    new_stations_data = []

    for i, point in enumerate(new_points):
        x, y = point
        square = Polygon([
            (x - half_size, y - half_size),
            (x + half_size, y - half_size),
            (x + half_size, y + half_size),
            (x - half_size, y + half_size),
            (x - half_size, y - half_size)
        ])

        centroid = Point(x, y)
        station_name = f"new_{i}"

        # Trouver le département correspondant au centre du carré
        departement = geo_france.loc[geo_france.contains(centroid), 'departement'].values
        departement = departement[0] if len(departement) > 0 else 'Unknown'

        new_zones_geom.append({'geometry': square, 'station': station_name, 'departement': departement, 'hauteurstation' : 0})
        new_stations_data.append({'geometry': centroid, 'station': station_name, 'set': 'test', 'departement': departement})

    new_zones_gdf = gpd.GeoDataFrame(new_zones_geom, crs=zones.crs)
    if zones is not None:
        zones_updated = zones.append(new_zones_gdf, ignore_index=True)
    else:
        zones_updated = new_zones_gdf

    new_stations_gdf = gpd.GeoDataFrame(new_stations_data, crs=stations.crs)
    if stations is not None:
        stations = stations.append(new_stations_gdf, ignore_index=True)
    else:
        stations = new_stations_gdf

    return zones_updated, stations

def add_new_zones_from_polygon(zones, new_polygons, stations, geo_france):
    """
    Ajoute de nouvelles zones sous forme de polygones et met à jour le GeoDataFrame des stations.

    Parameters:
    - zones (gpd.GeoDataFrame): Le GeoDataFrame existant contenant les zones.
    - new_polygons (list of shapely.geometry.Polygon): Liste des nouvelles zones sous forme de polygones.
    - stations (gpd.GeoDataFrame): GeoDataFrame des stations à mettre à jour.
    - geo_france (gpd.GeoDataFrame): GeoDataFrame contenant les départements de France.

    Returns:
    - zones_updated (gpd.GeoDataFrame): Le GeoDataFrame mis à jour avec les nouvelles zones.
    - stations (gpd.GeoDataFrame): Le GeoDataFrame des stations mis à jour avec les nouvelles stations.
    """

    new_zones_geom = []
    new_stations_data = []

    for i, polygon in enumerate(new_polygons):
        centroid = polygon.centroid
        station_name = f"new_{i}"

        # Trouver le département correspondant au centre du polygone
        departement = geo_france.loc[geo_france.contains(centroid), 'departement'].values
        departement = departement[0] if len(departement) > 0 else 'Unknown'

        new_zones_geom.append({'geometry': polygon, 'station': station_name, 'departement': departement, 'hauteurstation' : 0})
        new_stations_data.append({'geometry': centroid, 'station': station_name, 'set': 'test', 'departement': departement})

    new_zones_gdf = gpd.GeoDataFrame(new_zones_geom, crs=zones.crs)
    if zones is not None:
        zones_updated = zones.append(new_zones_gdf, ignore_index=True)
    else:
        zones_updated = new_zones_gdf

    new_stations_gdf = gpd.GeoDataFrame(new_stations_data, crs=stations.crs)
    if stations is not None:
        stations = stations.append(new_stations_gdf, ignore_index=True)
    else:
        stations = new_stations_gdf

    return zones_updated, stations

def create_zones(stations, zones, resolution, dir_output):
    logger.info(f'Create zones raster, resolution -> {resolution}')
    lats = resolutions[resolution]['y']
    longs = resolutions[resolution]['x']
    check_and_create_path(dir_output / 'database')
    ustations = stations.station.unique()
    for station in ustations:
        logger.info(f'################## {station} ################')
        station_gdf = zones[zones['station'] == station]
        dir_out = '.'
        try:
            data = rasterisation(station_gdf, lats, longs, column='hauteurstation', defval = 0, name='default', dir_output=dir_out)
            os.remove(Path(dir_out) / 'default.geojson')
            imshow_single_image(data, dir_output / 'database' / station, station)
            save_object(data, f'{station}.pkl', dir_output / 'database')
        except Exception as e:
            stations = stations[stations['station'] != station]
            zones = zones[zones['station'] != station]
            logger.info(f'{station} -> {e}')

    stations.to_file(dir_output / 'viginond_stations_clean.geojson', driver='GeoJSON')
    zones.to_file(dir_output / 'viginond_zip_clean.geojson', driver='GeoJSON')

    return stations, zones

def read_cosia(dir_data):
    cosia = gpd.read_file(dir_data / 'cosia' / 'cosia.geojson')

    # Modify the classes
    cosia = cosia[(cosia['numero'] != 7)]
    cosia.loc[cosia[cosia['numero'].isin([18, 2, 4])].index, 'numero'] = 1
    cosia.loc[cosia[cosia['numero'].isin([3, 5])].index, 'numero'] = 2
    cosia.loc[cosia[cosia['numero'].isin([6, 7])].index, 'numero'] = 3
    cosia.loc[cosia[cosia['numero'] == 8].index, 'numero'] = 4
    cosia.loc[cosia[cosia['numero'] == 10].index, 'numero'] = 5
    cosia.loc[cosia[cosia['numero'] == 9].index, 'numero'] = 6
    cosia.loc[cosia[cosia['numero'] == 15].index, 'numero'] = 7
    cosia.loc[cosia[cosia['numero'].isin([14, 17])].index, 'numero'] = 8

    return cosia

def read_elevation(dir_data):
    elevation = pd.read_csv(dir_data / 'elevation' / 'elevation.csv')
    elevation = gpd.GeoDataFrame(elevation, geometry=gpd.points_from_xy(elevation.longitude, elevation.latitude))
    return elevation

def read_foret(dir_data):
    foret = gpd.read_file(dir_data / 'BDFORET' / 'foret.geojson')
    return foret

def raster_cosia(feature_data, lats, longs, dir_out):
    """
    Fonction de rasterisation pour les données COSIA avec masques binaires pour chaque valeur unique.

    Parameters:
    - feature_data (GeoDataFrame): Données géographiques à rasteriser.
    - lats (int): Nombre de pixels dans la direction latitudinale.
    - longs (int): Nombre de pixels dans la direction longitudinale.
    - dir_out (str or Path): Répertoire de sortie pour le fichier raster.
    
    Returns:
    - np.ndarray: Tableau 3D contenant les masques binaires pour chaque valeur unique.
    """
    # Rasterisation des données
    data = rasterisation(feature_data, lats, longs, column='numero', defval=-1, name='default', dir_output=dir_out)
    
    # Obtenir les valeurs uniques dans le raster
    unique_values = np.unique(data)
    
    # Exclure la valeur par défaut (-1) si elle existe
    unique_values = unique_values[unique_values != -1]
    
    # Initialiser un tableau 3D pour stocker les masques
    masks = np.zeros((len(unique_values), data.shape[0], data.shape[1]), dtype=np.uint8)
    
    # Créer un masque pour chaque valeur unique
    for i, val in enumerate(unique_values):
        masks[i] = (data == val).astype(np.uint8)
    
    return masks

def raster_foret(feature_data, lats, longs, dir_out):
    """
    Fonction de rasterisation pour les données de forêt avec masques binaires pour chaque valeur unique.

    Parameters:
    - feature_data (GeoDataFrame): Données géographiques à rasteriser.
    - lats (int): Nombre de pixels dans la direction latitudinale.
    - longs (int): Nombre de pixels dans la direction longitudinale.
    - dir_out (str or Path): Répertoire de sortie pour le fichier raster.
    
    Returns:
    - np.ndarray: Tableau 3D contenant les masques binaires pour chaque valeur unique.
    """
    # Rasterisation des données
    data = rasterisation(feature_data, lats, longs, column='code', defval=-1, name='default', dir_output=dir_out)
    
    # Obtenir les valeurs uniques dans le raster
    unique_values = np.unique(data)
    
    # Exclure la valeur par défaut (-1) si elle existe
    unique_values = unique_values[unique_values != -1]
    
    # Initialiser un tableau 3D pour stocker les masques
    masks = np.zeros((len(unique_values), data.shape[0], data.shape[1]), dtype=np.uint8)
    
    # Créer un masque pour chaque valeur unique
    for i, val in enumerate(unique_values):
        masks[i] = (data == val).astype(np.uint8)
    
    return masks

def switch_read_features(feature, dir_data):
    if feature == 'cosia':
        return read_cosia(dir_data)
    elif feature == 'elevation':
        return read_elevation(dir_data)
    elif feature == 'foret':
        return read_foret(dir_data)
    else:
        raise ValueError(f'{feature} unknowed feature')

def switch_raster_features(feature, feature_data, lats, longs):
    dir_out = '.'
    try:
        if feature == 'cosia':
            data = raster_cosia(feature_data, lats, longs, dir_out)
        elif feature == 'elevation':
            data = rasterisation(feature_data, lats, longs, column='altitude', defval=-1, name='default', dir_output=dir_out)
            data = data[np.newaxis, :]
        elif feature == 'foret':
            data = raster_foret(feature_data, lats, longs, dir_out)
        else:
            raise ValueError(f'{feature} unknowed feature') 
        
        os.remove(Path(dir_out) / 'default.geojson')
        return data
    except Exception as e:
        logger.info(e)

def create_features(stations, zones, resolution, features, dir_output):
    logger.info(f'Create {features} raster, resolution -> {resolution}')

    lats = resolutions[resolution]['y']
    longs = resolutions[resolution]['x']
    check_and_create_path(dir_output / 'database')
    ustations = stations.station.unique()

    stations.sort_values('departement', inplace=True)
    last_departement = None
    
    for station in ustations:
        logger.info(f'################## {station} ################')

        data = None
        departement = stations[stations['station'] == station]['departement'].values[0]
        root_data = root_features / departement / 'data'
        
        station_zones = zones[zones['station'] == station]
        
        if last_departement != departement:
            data_per_departement = {}

        for feature in features:
            if feature not in data_per_departement.keys():
                feature_data = switch_read_features(feature, dir_data=root_data)
                data_per_departement[feature] = feature_data
            else:
                feature_data = data_per_departement[feature]
            
            minx, miny, maxx, maxy = station_zones.total_bounds  # Extraction des bornes
            bbox_zone = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=feature_data.crs)

            feature_data_crop = gpd.overlay(feature_data, bbox_zone, how='intersection')

            if data is None:
                # Rasteriser et enregistrer les données
                data = switch_raster_features(feature, feature_data_crop, lats, longs)
                for i in range(data.shape[0]):
                    imshow_single_image(data[i], dir_output / 'database' / station, f'{feature}_{i}')
            else:
                data_ = switch_raster_features(feature, feature_data_crop, lats, longs)
                for i in range(data.shape[0]):
                    imshow_single_image(data_[i], dir_output / 'database' / station, f'{feature}_{i}')
                data = np.concatenate((data, data_), axis=0)

        # Sauvegarder l'objet en pickle
        save_object(data, f'{station}_features.pkl', dir_output / 'database')
        last_departement = departement

def create_database(dir_input, dir_output, resolution, features, doFeatures, doZones):
    dir_output = dir_output
    check_and_create_path(dir_output)

    ############################## Zones ##############################

    if doZones:
        stations, zones = read_stations_zones(dir_input)
        stations, zones = create_zones(stations, zones, resolution, dir_output)
    else:
        stations = gpd.read_file(dir_output / 'viginond_stations_clean.geojson')
        zones = gpd.read_file(dir_output / 'viginond_zip_clean.geojson')

    ############################## Features ##############################

    if doFeatures:
        create_features(stations, zones, resolution, features, dir_output)

    return stations, zones

def load_new_departement_data():
    geo_france = gpd.read_file(root_features / 'france' / 'data' / 'geo' / 'departements.geojson')

    ################################################################ Ain ############################################
    geo = gpd.read_file(root_features / 'departement-01-ain' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(None, geo, None, geo_france)

    ################################################################ Hautes-Alpes ##################################
    geo = gpd.read_file(root_features / 'departement-05-hautes-alpes' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Alpes-Maritimes ################################
    geo = gpd.read_file(root_features / 'departement-06-alpes-maritimes' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Cantal #########################################
    geo = gpd.read_file(root_features / 'departement-15-cantal' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Corse ###########################################
    geo = gpd.read_file(root_features / 'departement-20-corse' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Doubs ##########################################
    geo = gpd.read_file(root_features / 'departement-25-doubs' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Eure-et-Loir ###################################
    geo = gpd.read_file(root_features / 'departement-28-eure-et-loir' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Hérault ########################################
    geo = gpd.read_file(root_features / 'departement-34-herault' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Isère ###########################################
    geo = gpd.read_file(root_features / 'departement-38-isere' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Moselle ########################################
    geo = gpd.read_file(root_features / 'departement-57-moselle' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Oise ###########################################
    geo = gpd.read_file(root_features / 'departement-60-oise' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Haute-Saône ####################################
    geo = gpd.read_file(root_features / 'departement-70-haute-saone' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Savoie #########################################
    geo = gpd.read_file(root_features / 'departement-73-savoie' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Somme ##########################################
    geo = gpd.read_file(root_features / 'departement-80-somme' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Var ############################################
    geo = gpd.read_file(root_features / 'departement-83-var' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Territoire de Belfort ##########################
    geo = gpd.read_file(root_features / 'departement-90-territoire-de-belfort' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    ################################################################ Hauts-de-Seine #################################
    geo = gpd.read_file(root_features / 'departement-92-hauts-de-seine' / 'data' / 'geo' / 'departement.geojson')
    zones, stations = add_new_zones_from_polygon(zones, geo, stations, geo_france)

    return zones, stations