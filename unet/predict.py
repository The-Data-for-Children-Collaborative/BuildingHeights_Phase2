import torch
import torchvision
from PIL import Image

import net

if __name__ == '__main__':

    # Check for MPS/CUDA

    if torch.backends.mps.is_available() == True:
        device = torch.device('mps')
    elif torch.cuda.is_available() == True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Device: {}'.format(device))

    # Parameters
    rgb_name = 'train/4331-BHM/a.tif'
    prediction_name = 'predict.png'
    weights_dict = 'snapshot_1.pth.tar'

    # The model
    model = net.unet()
    model = model.to(device)

    # The trained weights are loaded
    state_dict = torch.load(weights_dict)['state_dict']
    model.load_state_dict(state_dict)

    # The model is set to evaluation mode
    model.eval()

    rgb = Image.open(rgb_name)
    rgb = torchvision.transforms.functional.pil_to_tensor(rgb)

    # We add a dummy dimension, creating a size-1 batch
    rgb = rgb[None, :]

    # The image is resized to 256 x 256
    rgb = torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear')

    # The model is run on the input image
    rgb = rgb.to(device)
    output = model(rgb)
    output = torch.nn.functional.interpolate(output, size=(500,500), mode='bilinear')

    # At last we save the prediction
    predicted_image = torchvision.transforms.functional.to_pil_image(output[0])
    predicted_image.save(prediction_name)
