from dataloader import *

def imshow_single_image(data, dir_output, out_name, cmap='jet', figsize=(15,5)):
    check_and_create_path(dir_output)
    plt.figure(figsize=figsize)
    plt.imshow(data, cmap=cmap)
    plt.savefig(dir_output / f'{out_name}.png')
    plt.close('all')