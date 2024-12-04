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
parser.add_argument('-r', '--resolution', type=str, help='resolution')

args = parser.parse_args()

name_exp = args.name
doZones = args.doZones
doFeatures = args.doFeatures
resolution = args.resolution

features = ['cosia', 'elevation']

###################################################################################

dir_output = root_script / name_exp
check_and_create_path(dir_output)
dir_input = root_script / 'stations'

stations, zones = read_stations_zones(dir_input)

if doZones or doFeatures:
    create_database(stations, zones, dir_output, resolution, features, doFeatures, doZones)

train_zone, val_zone, test_zone = split_zones(stations, zones, 0.15, 0.2)

train_loader, val_loader, test_loader = create_train_val_test_loader(train_zone, val_zone, test_zone, device, dir_output)

############################################################# UNET 1 ##################################################

model_config = {'lr' : 0.001,
                'epoch' : 1000,
                'PATIENCE_CNT' : 200,
                'CHECKPOINT' : 25,
                'loss_name' : 'rmse',
                'model_name' : 'unet',
                'dir_output' : dir_output}

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

wrapped_train(model, train_loader, val_loader, model_config)

wrapped_test(model, train_loader, val_loader, model_config)