import socket
import torch

def get_machine_info():
    try:
        # Obtenir le nom d'hôte de la machine
        hostname = socket.gethostname()
        print(f"Nom de l'hôte : {hostname}")

        # Obtenir l'adresse IP locale
        local_ip = socket.gethostbyname(hostname)
        print(f"Adresse IP locale : {local_ip}")
    except Exception as e:
        print(f"Une erreur est survenue : {e}")

    return hostname

is_pc = get_machine_info() == 'caron-Precision-7780'

MLFLOW = False
if is_pc:
    MLFLOW = False
if MLFLOW:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

resolutions = {'2x2' : {'x' : 0.02875215641173088,'y' :  0.020721094073767096},
                '1x1' : {'x' : 0.01437607820586544,'y' : 0.010360547036883548},
                '0.5x0.5' : {'x' : 0.00718803910293272,'y' : 0.005180273518441774},
                '0.03x0.03' : {'x' : 0.0002694945852326214,'y' :  0.0002694945852352859}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # The device on which we train each models
