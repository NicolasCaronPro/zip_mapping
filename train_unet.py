from train import *
import argparse

parser = argparse.ArgumentParser(
    prog='Train',
    description='',
)

####################################### INPUT ####################################

parser.add_argument('-n', '--name', type=str, help='Name of the experiment')
parser.add_argument('-dz', '--doZones', type=str, help='doZones')
parser.add_argument('-df', '--doFeatures', type=str, help='doFeatures')
parser.add_argument('-dl', '--doLoader', type=str, help='doLoader')
parser.add_argument('-r', '--resolution', type=str, help='resolution')
parser.add_argument('-train', '--train', type=str, help='Do train')
parser.add_argument('-test', '--test', type=str, help='Do test')

args = parser.parse_args()

name_exp = args.name
doZones = args.doZones == 'True'
doFeatures = args.doFeatures == 'True'
resolution = args.resolution
doTest = args.test == 'True'
doTrain = args.train == 'True'
doLoader = args.doLoader == 'True'

features = ['foret']

###################################################################################

dir_output = root_script / name_exp
check_and_create_path(dir_output)
dir_input = root_script / 'stations'

if doZones or doFeatures:
    stations, zones = create_database(dir_input, dir_output, resolution, features, doFeatures, doZones)

if doLoader:
    if 'stations' not in locals():
        stations = gpd.read_file(dir_output / 'database' / 'viginond_stations_clean.geojson')
        zones = gpd.read_file(dir_output / 'database' / 'viginond_zip_clean.geojson')
    train_zone, val_zone, test_zone = split_zones(stations, zones, 0.15, 0.2)
    train_loader, val_loader, test_loader = create_train_val_test_loader(train_zone, val_zone, test_zone, device, dir_output)
else:
    train_loader, val_loader, test_loader = load_train_val_test_loader(device, dir_output)

############################################################# UNET 1 ##################################################

model_name = 'unet' 

model_config = {'lr' : 0.001,
                'epochs' : 1000,
                'PATIENCE_CNT' : 200,
                'CHECKPOINT' : 25,
                'loss_name' : 'rmse',
                'model_name' : 'unet',
                'dir_output' : dir_output / 'model'}

default_params = {
            'n_channels': len(features),
            'n_classes': 1,
            'features': [64, 128, 256, 512, 1024],
            'bilinear': False,
            'device': device

        }

model = UNet(
    n_channels=default_params['n_channels'],
    n_classes=default_params['n_classes'],
    features=default_params['features'],
    bilinear=default_params['bilinear']
).to(default_params['device'])

model_config.update(default_params)

save_object(model_config, f'{model_name}_config.pkl', dir_output / 'model')

if doTrain:
    wrapped_train(model, train_loader, val_loader, model_config)
else:
    model_config = read_object(f'{model_name}_config.pkl', dir_output / 'model')
    model.load_state_dict(torch.load(dir_output / 'model.pt', map_location=device, weights_only=True), strict=False)

if doTest:
    wrapped_test(model, model_config, test_loader, model_config)