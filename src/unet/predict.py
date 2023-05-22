import torch
import torchvision
from PIL import Image
import numpy as np
import shutil

import net

def save_as_grayscale(ar, filename):

    ar = ar * 256
    ar = np.clip(ar, 0, 255)
    ar = ar.astype(np.uint8)

    img = Image.fromarray(ar)
    img = img.convert('L')
    img = img.quantize(colors=256)
    img.save(filename)


def do_prediction(rgb_name, target_name, prediction_name, reprocessed_rgb_name=None, reprocessed_target_name=None):

    # Check for MPS/CUDA

    if torch.backends.mps.is_available() == True:
        device = torch.device('mps')
    elif torch.cuda.is_available() == True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Device: {}'.format(device))

    # Parameters
    weights_dict = 'snapshot_120.pth.tar'

    # The vegetation threshold must match the one used in training
    vegetation_threshold = 6

    # The model
    model = net.unet()
    model = model.to(device)

    # The trained weights are loaded
    state_dict = torch.load(weights_dict, map_location=torch.device(device))['state_dict']
    model.load_state_dict(state_dict)

    # The model is set to evaluation mode
    model.eval()

    # We load the input image
    rgb = Image.open(rgb_name)
    rgb = torchvision.transforms.functional.pil_to_tensor(rgb)
    rgb = rgb.float().div(256)
    rgb = rgb[:3]

    # We add a dummy dimension, creating a size-1 batch
    rgb = rgb[None, :]

    # The image is resized to 256 x 256
    rgb = torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear')

    # The model is run on the input image
    rgb = rgb.to(device)
    output = model(rgb)

    output = torch.nn.functional.interpolate(output, size=(500,500), mode='bilinear')
    output = output[0,0]

    target = Image.open(target_name)
    target = torchvision.transforms.functional.pil_to_tensor(target)
    target.apply_(lambda x: 1 if x > vegetation_threshold else 0)
    target = target[0]

    ar1 = output.detach().cpu().numpy()
    ar2 = np.array(target)

    save_as_grayscale(ar1, prediction_name)

    if reprocessed_target_name != None:
        save_as_grayscale(ar2, reprocessed_target_name)

    if reprocessed_rgb_name != None:
        shutil.copy(rgb_name, reprocessed_rgb_name)


if __name__ == '__main__':

    idlists = [['3313', '333', '6'],
               ['3313', '461', '8'],
               ['3313', '421', '11'],
               ['3313', '143', '11'],
               ['3313', '433', '9'],
               ['3313', '151', '7'],
               ['3313', '162', '6'],
               ['3313', '123', '7'],
               ['3313', '424', '16'],
               ['3313', '153', '15'],
               ['3313', '163', '10']]

    counter = 0

    for idlist in idlists:

        id1 = idlist[0]
        id2 = idlist[1]
        id3 = idlist[2]

        print('{} {} {}'.format(id1, id2, id3))

        maxar_filename = id1 + '/maxar_' + id1 + '_' + id2 + '_s' + id3 + '.tif'
        vhm_filename = id1 + '/vhm_' + id1 + '_' + id2 + '_s' + id3 + '.tif'

        do_prediction(maxar_filename, vhm_filename, 'output' + str(counter) + '.png', 'input'  + str(counter) + '.tif', 'reference'  + str(counter) +  '.png')
        counter += 1
