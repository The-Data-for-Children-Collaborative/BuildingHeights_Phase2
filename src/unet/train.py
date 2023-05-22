import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import time

import net
import utils

def do_training_epoch(train_loader, vegetation_threshold, losses, end):

        for i, sample_batched in enumerate(train_loader):
        
            rgb, vhm = sample_batched['rgb'], sample_batched['vhm']

            # The images in each training pair are resized to 256 x 256

            rgb = torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear')
            vhm = torch.nn.functional.interpolate(vhm, size=(256, 256), mode='bilinear')

            # We convert the VHM to a binary map, with 0's where the vegetation exceeds the threshold
            # and 1's in the areas that we actually want to be considered when training

            vhm.apply_(lambda x: 1 if x > vegetation_threshold else 0)

            rgb = rgb.to(device)
            vhm = vhm.to(device, non_blocking=True)

            # We initialize the gradients to zero
            optimizer.zero_grad()

            # The model is evaluated on the current sample
            output = model(rgb)

            # We calculate the loss and update the loss counter
            loss = torch.log(torch.abs(output - vhm) + 0.5).mean()
            losses.update(loss.data, rgb.size(0))

            # Then we backpropagate to calculate the gradients, and we run one optimizer step
            loss.backward()
            optimizer.step()

            # Bookkeeping: calculating elapsed time, printing some useful info

            batch_time.update(time.time() - end)
            end = time.time()

            message = 'Epoch: [{0}][{1}/{2}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t' \
                      'Loss {losses.val:.4f} ({losses.avg:.4f})' \
                       .format(epoch, i, len(train_loader), batch_time=batch_time, losses=losses)

            print(message)


def loss_on_test_set(test_loader, vegetation_threshold):

    '''
        Given a (trained or partially trained) model, evaluates the loss on the training set.

        Arguments: test_loader, a torch.utils.data.DataLoader object helping us to load the test pairs
    '''

    total_loss = 0
    counter = 0

    for i, sample_batched in enumerate(test_loader):

        rgb, vhm = sample_batched['rgb'], sample_batched['vhm']

        # The images in each training pair are resized to 256 x 256

        rgb = torch.nn.functional.interpolate(rgb, size=(256, 256), mode='bilinear')
        vhm = torch.nn.functional.interpolate(vhm, size=(256, 256), mode='bilinear')

        # We convert the VHM to a binary map, with 0's where the vegetation exceeds the threshold
        # and 1's in the areas that we actually want to be considered when training

        vhm.apply_(lambda x: 1 if x > vegetation_threshold else 0)

        rgb = rgb.to(device)
        vhm = vhm.to(device, non_blocking=True)

        # The model is evaluated on the current sample
        output = model(rgb)

        # We calculate the loss and update the loss counter
        loss = torch.log(torch.abs(output - vhm) + 0.5).mean()
        
        total_loss += loss.detach().cpu()
        counter += 1

    return total_loss / counter


if __name__ == '__main__':

    # Check for MPS/CUDA

    if torch.backends.mps.is_available() == True:
        device = torch.device('mps')
    elif torch.cuda.is_available() == True:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print('Device: {}'.format(device))

    # Various configuration options
    nr_epochs = 30
    vegetation_threshold = 6
    csv_train_file = 'train.csv'
    csv_test_file = 'test.csv'
    snapshot_path = 'snapshot_'

    # Hyperparameters
    initial_lr = 1e-4
    weight_decay = 1e-4

    # Data loader
    # One should check carefully num_workers and pin_memory,
    # these are important when using CUDA.
    train_loader = DataLoader(utils.myDataset(csv_file=csv_train_file), batch_size=16, num_workers=0, pin_memory=False)
    test_loader = DataLoader(utils.myDataset(csv_file=csv_test_file), batch_size=16, num_workers=0, pin_memory=False)

    # The model
    model = net.unet()
    model = model.to(device)

    # We count the number of trainable parameters in the model and we print it
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters in the current model: {}'.format(total_params))

    # Enables the cuDNN autotuner
    cudnn.benchmark = True

    # Selects the Adam optimizer, with parameters as specified
    optimizer = torch.optim.Adam(model.parameters(), initial_lr, weight_decay=weight_decay)

    # Iterate over all epocs
    for epoch in range(1, nr_epochs + 1):

        batch_time = utils.AverageMeter()
        losses = utils.AverageMeter()

        # The learning rate is adjusted for this epoch

        lr = initial_lr * (0.9 ** (epoch // 5))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # The model is set to training mode
        model.train()

        # We measure the time starting from here
        end = time.time()

        do_training_epoch(train_loader, vegetation_threshold, losses, end)

        # At the end of each epoch we save the trained weights
        out_name = snapshot_path + str(epoch) + '.pth.tar'
        state = {'state_dict': model.state_dict()}
        torch.save(state, out_name)
        print('Snapshot saved to: {}'.format(out_name))

        # And we also calculate the loss on the test set
        print('Average loss on training set at epoch {}: {}'.format(epoch, losses.avg))

        lots = loss_on_test_set(test_loader, vegetation_threshold)
        print('Average loss on test set at epoch {}: {}'.format(epoch, lots))
