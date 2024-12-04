from arborescence import *
from config import *
import numpy as np
import geopandas as gpd
import pandas as pd
import logging
import pickle
from pathlib import Path
import torch
import os
from osgeo import gdal, ogr
import rasterio
import rasterio.features
import rasterio.warp
import sys
import random
import matplotlib.pyplot as plt
import math

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")

# Handler pour afficher les logs dans le terminal
streamHandler = logging.StreamHandler(stream=sys.stdout)
streamHandler.setFormatter(logFormatter)
logger.addHandler(streamHandler)

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def save_object(obj, filename: str, path : Path):
    check_and_create_path(path)
    with open(path / filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def save_object_torch(obj, filename : str, path : Path):
    check_and_create_path(path)
    torch.save(obj, path/filename)

def read_object(filename: str, path : Path):
    if not (path / filename).is_file():
        logger.info(f'{path / filename} not found')
        return None
    return pickle.load(open(path / filename, 'rb'))

def read_tif(name):
    """
    Open a satellite images and return bands, latitude and longitude of each pixel.
    """
    with rasterio.open(name) as src:
        dt = src.read()
        height = dt.shape[1]
        width = dt.shape[2]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons = np.array(xs)
        lats = np.array(ys)
        src.close()
    return dt, lons, lats

def rasterisation(h3, lats, longs, column='cluster', defval = 0, name='default', dir_output='/media/caron/X9 Pro/corbeille'):
    #h3['cluster'] = h3.index

    h3.to_file(dir_output + '/' + name+'.geojson', driver='GeoJSON')

    input_geojson = dir_output + '/' + name+'.geojson'
    output_raster = dir_output + '/' + name+'.tif'

    # Si on veut rasteriser en fonction de la valeur d'un attribut du vecteur, mettre son nom ici 
    attribute_name = column

    # Taille des pixels
    if isinstance(lats, float):
        pixel_size_y = lats
        pixel_size_x = longs
    else:
        pixel_size_x = abs(longs[0][0] - longs[0][1])
        pixel_size_y = abs(lats[0][0] - lats[1][0])
    
    #pixel_size_x = res[dim][0]
    #pixel_size_y = res[dim][1]

    source_ds = ogr.Open(input_geojson)
    source_layer = source_ds.GetLayer()

    # On obtient l'étendue du raster
    x_min, x_max, y_min, y_max = source_layer.GetExtent()
    # On calcule le nombre de pixels
    width = int((x_max - x_min) / pixel_size_x)
    height = int((y_max - y_min) / pixel_size_y)

    # Oncrée un nouveau raster dataset et on passe de "coordonnées image" (pixels) à des coordonnées goréférencées
    driver = gdal.GetDriverByName('GTiff')
    output_ds = driver.Create(output_raster, width, height, 1, gdal.GDT_Float32)
    output_ds.GetRasterBand(1).Fill(defval)
    output_ds.SetGeoTransform([x_min, pixel_size_x, 0, y_max, 0, -pixel_size_y])
    output_ds.SetProjection(source_layer.GetSpatialRef().ExportToWkt())

    if attribute_name != '' :
        # On  rasterise en fonction de l'attribut donné
        gdal.RasterizeLayer(output_ds, [1], source_layer, options=["ATTRIBUTE=" + attribute_name])
    else :
        # On  rasterise. Le raster prend la valeur 1 là où il y a un vecteur
        gdal.RasterizeLayer(output_ds, [1], source_layer)

    output_ds = None
    source_ds = None

    res, _, _ = read_tif(dir_output + '/' + name+'.tif')
    os.remove(dir_output + '/' + name+'.tif')
    return res[0]

def get_existing_run(run_name):
    # Récupère tous les runs avec le nom spécifié
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=['0'],  # Spécifiez ici l'ID de l'expérience si nécessaire
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        run_view_type=mlflow.entities.ViewType.ALL
    )
    
    # Si un run est trouvé, le retourner
    if runs:
        return runs[0]  # retourne le premier run trouvé
    return None

def resize_no_dim(input_image, height, width):
    """
    Resize the input_image into heigh, with, dim
    """
    img = img_as_float(input_image)
    img = transform.resize(img, (height, width), mode='constant', order=0,
                 preserve_range=True, anti_aliasing=True)
    return np.asarray(img)

def remove_accents(input_str):
    """
    Retire tous les accents d'une chaîne de caractères.

    Parameters:
    - input_str (str): La chaîne de caractères d'entrée.

    Returns:
    - str: La chaîne de caractères sans accents.
    """
    import unicodedata
    # Décompose les caractères accentués en caractères de base + marques d'accent
    nfkd_form = unicodedata.normalize('NFKD', input_str)

    # Filtre uniquement les caractères de base (catégorie 'Mn' = Nonspacing Mark)
    return ''.join(c for c in nfkd_form if not unicodedata.combining(c))
