from database import *
from torch import optim
import tqdm

def launch_train_loader(model, loader,
                  criterion, optimizer):
    
    model.train()
    for i, data in enumerate(loader, 0):

        inputs, target = data

        output = model(inputs)

        target = target.view(output.shape)
        weights = weights.view(output.shape)
        
        target = torch.masked_select(target, weights.gt(0))
        output = torch.masked_select(output, weights.gt(0))
        weights = torch.masked_select(weights, weights.gt(0))
        loss = criterion(output, target, weights)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss

def launch_val_test_loader(model, loader, criterion):
    """
    Evaluates the model using the provided data loader, with optional autoregression.

    Note:
    - 'node_id' and 'date_id' are not included in features_name but can be found in labels[:, 0] and labels[:, 4].

    Parameters:
    - model: The PyTorch model to evaluate.
    - loader: DataLoader providing the validation or test data.
    - features: List or tuple of feature indices to use.
    - target_name: Name of the target variable.
    - criterion: Loss function.
    - optimizer: Optimizer (not used during evaluation but included for consistency).
    - autoRegression: Boolean indicating whether to use autoregression.
    - features_name: List of feature names.
    - hybrid: Boolean indicating if the model uses hybrid inputs.

    Returns:
    - total_loss: The cumulative loss over the dataset.
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for i, data in enumerate(loader, 0):
            
            inputs, labels = data

            output = model(inputs)

            target = labels.view(output.shape)
            weights = weights.view(output.shape)

            # Mask out invalid weights
            valid_mask = weights.gt(0)
            target = torch.masked_select(target, valid_mask)
            output = torch.masked_select(output, valid_mask)
            weights = torch.masked_select(weights, valid_mask)
            loss = criterion(output, target, weights)
            loss = criterion(output, target)

            total_loss += loss.item()

    return total_loss

def func_epoch(model, train_loader, val_loader, optimizer, criterion):
    
    train_loss = launch_train_loader(model, train_loader, criterion, optimizer)

    if val_loader is not None:
        val_loss = launch_val_test_loader(model, val_loader, criterion)
    
    else:
        val_loss = train_loss

    return val_loss, train_loss

# Fonction pour sélectionner la fonction de perte via son nom
def get_loss_function(loss_name):
    # Dictionnaire pour associer le nom de la fonction de perte à sa classe correspondante
    loss_dict = {
        "poisson": PoissonLoss,
        "rmsle": RMSLELoss,
        "rmse": RMSELoss,
        "mse": MSELoss,
        "huber": HuberLoss,
        "logcosh": LogCoshLoss,
        "tukeybiweight": TukeyBiweightLoss,
        "exponential": ExponentialLoss,
        "weightedcrossentropy": WeightedCrossEntropyLoss
    }
    loss_name = loss_name.lower()
    if loss_name in loss_dict:
        return loss_dict[loss_name]()
    else:
        raise ValueError(f"Loss function '{loss_name}' not found in loss_dict.")

def plot_train_val_loss(epochs, train_loss_list, val_loss_list, dir_output):
    # Création de la figure et des axes
    plt.figure(figsize=(10, 6))

    # Tracé de la courbe de val_loss
    plt.plot(epochs, val_loss_list, label='Validation Loss', color='blue')

    # Ajout de la légende
    plt.legend()

    # Ajout des labels des axes
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Ajout d'un titre
    plt.title('Validation Loss over Epochs')
    plt.savefig(dir_output / 'Validation.png')
    plt.close('all')

    # Tracé de la courbe de train_loss
    plt.plot(epochs, train_loss_list, label='Training Loss', color='red')

    # Ajout de la légende
    plt.legend()

    # Ajout des labels des axes
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    # Ajout d'un titre
    plt.title('Training Loss over Epochs')
    plt.savefig(dir_output / 'Training.png')
    plt.close('all')

def wrapped_train(model, train_loader, val_loader, params):
    lr = params['lr']
    epoch = params['epoch']
    PATIENCE_CNT = params['PATIENCE_CNT']
    CHECKPOINT = params['CHECKPOINT']
    epochs = params['epochs']
    loss_name = params['loss_name']
    model_name = params['model_name']
    dir_output = params['dir_output']

    if MLFLOW:
        existing_run = get_existing_run(f'{model_name}_')
        if existing_run:
            mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
        else:
            mlflow.start_run(run_name=f'{model_name}_', nested=True)

    assert train_loader is not None and val_loader is not None

    check_and_create_path(dir_output)

    criterion = get_loss_function(loss_name)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    BEST_VAL_LOSS = math.inf
    BEST_MODEL_PARAMS = None
    patience_cnt = 0

    val_loss_list = []
    train_loss_list = []
    epochs_list = []

    logger.info('Train model with')
    for epoch in tqdm(range(epochs)):
        val_loss, train_loss = func_epoch(model, train_loader, val_loader, optimizer, criterion)
        train_loss = train_loss.item()
        val_loss = round(val_loss, 3)
        train_loss = round(train_loss, 3)
        val_loss_list.append(val_loss)
        train_loss_list.append(train_loss)
        epochs_list.append(epoch)
        if val_loss < BEST_VAL_LOSS:
            BEST_VAL_LOSS = val_loss
            BEST_MODEL_PARAMS = model.state_dict()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE_CNT:
                logger.info(f'Loss has not increased for {patience_cnt} epochs. Last best val loss {BEST_VAL_LOSS}, current val loss {val_loss}')
                save_object_torch(model.state_dict(), 'last.pt', dir_output)
                save_object_torch(BEST_MODEL_PARAMS, 'best.pt', dir_output)
                plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, dir_output)
                if MLFLOW:
                    mlflow.end_run()
                return
        if MLFLOW:
            mlflow.log_metric('loss', val_loss, step=epoch)
        if epoch % CHECKPOINT == 0:
            logger.info(f'epochs {epoch}, Val loss {val_loss}')
            logger.info(f'epochs {epoch}, Best val loss {BEST_VAL_LOSS}')
            save_object_torch(model.state_dict(), str(epoch)+'.pt', dir_output)

    logger.info(f'Last val loss {val_loss}')
    save_object_torch(model.state_dict(), 'last.pt', dir_output)
    save_object_torch(BEST_MODEL_PARAMS, 'best.pt', dir_output)
    plot_train_val_loss(epochs_list, train_loss_list, val_loss_list, dir_output)


def wrapped_test(model, model_config, test_loader, dir_output):
    with torch.no_grad():
        mae_func = torch.nn.L1Loss(reduce='none')
        mae = 0
        for data in test_loader:
            names, X, y = data
            
            logger.info(f'{torch.max(X)}')
            output = model(X, None)
        
            for b in range(y.shape[0]):
                loss = mae_func(output[b, 0], y[b, :, :, -1])
                yb = y[b].detach().cpu().numpy()
                outputb = output[b].detach().cpu().numpy()

                logger.info(f'loss {itest} {names[itest]}: {loss.item()} {np.max(outputb)}')

                if MLFLOW:
                    existing_run = get_existing_run(f'susceptibility_{names[b]}_{model_config["model_name"]}')
                    if existing_run:
                        mlflow.start_run(run_id=existing_run.info.run_id, nested=True)
                    else:
                        mlflow.start_run(run_name=f'susceptibility_{names[b]}_{model_config["model_name"]}', nested=True)
                    
                    mlflow.log_metric('MAE', loss.item())

                mae += loss.item()

                fig, ax = plt.subplots(1, 2, figsize=(15,5))
                raster = read_object(f'{names[b]}.pkl', dir_output / 'database')
                assert raster is not None
                raster = raster[0]
                
                output_image = resize_no_dim(outputb[0], raster.shape[0], raster.shape[1])
                y_image = resize_no_dim(yb[:,  :, -1], raster.shape[0], raster.shape[1])

                output_image[np.isnan(raster)] = np.nan
                y_image[np.isnan(raster)] = np.nan

                maxi = max(np.nanmax(output_image), np.nanmax(y_image))

                ax[0].set_title('Prediction map')
                ax[0].imshow(output_image, vmin=0, vmax=maxi)
                ax[1].set_title('Ground truth')
                ax[1].imshow(y_image, vmin=0, vmax=maxi)
                plt.tight_layout()
                plt.savefig(dir_output / 'test' / f'{names[b]}')
                plt.close('all')
                if MLFLOW:
                    mlflow.log_figure(fig, f'_{names[b]}.png')
                    mlflow.end_run()
                    
                itest += y.shape[0]

        logger.info(f'MAE on test set : {mae / itest}')