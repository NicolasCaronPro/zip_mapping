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
parser.add_argument('-mn', '--model_name', type=str, help='Model Name')

args = parser.parse_args()

name_exp = args.name
doZones = args.doZones
doFeatures = args.doFeatures
resolution = args.resolution
doNewTest = args.newTest
doTest = args.test
doTrain = args.train
model_name = args.model_name

###################################################################################

dir_output = root_script / name_exp
check_and_create_path(dir_output)
dir_input = root_script / 'stations'

features = ['cosia', 'elevation']

if doZones or doFeatures:
    stations, zones = load_new_departement_data()
    create_database(stations, zones, dir_output, resolution, features, doFeatures, doZones)
    train_loader, val_loader, test_loader = create_new_test_loader(zones, device, dir_output)
else:
    test_loader = load_new_test_loader(device, dir_output)

model_config = read_object(f'{model_name}_config.pkl', dir_output / 'model')

model = UNet(
    n_channels=model_config['n_channels'],
    n_classes=model_config['n_classes'],
    features=model_config['features'],
    bilinear=model_config['bilinear']
).to(model_config['device'])

model.load_state_dict(torch.load(dir_output / 'model', map_location=device, weights_only=True), strict=False)

wrapped_test(model, model_config, test_loader, model_config)