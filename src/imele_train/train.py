import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision

import numpy as np
import pandas as pd
from PIL import Image

import os
import sys
import time
import argparse

from models import modules, net, resnet, densenet, senet
import sobel
from transforms import *

class depthDataset(Dataset):
    '''
        Our super-simple data loader, an implementation of an torch.utils.data.DataLoader object
    '''
    def __init__(self, csv_file, transform=None):
        self.frame = pd.read_csv(csv_file, header=None)
        self.transform = transform

    def __getitem__(self, idx):

        image_name = self.frame.loc[idx, 0]
        depth_name = self.frame.loc[idx, 1]

        # convert numpy arrays to tifs
        _, image_extension = os.path.splitext(image_name)
        _, depth_extension = os.path.splitext(depth_name)        

        # If the extension of file is .npy we treat is as a numpy array
        # Otherwise, we treat is an image, and Pillow will take care of it.

        if image_extension == '.npy':
            image = np.load(image_name)
        else:
            image = Image.open(image_name)

        if depth_extension == '.npy':
            depth = np.load(depth_name)
            depth = depth.reshape(depth.shape[0], depth.shape[1], 1)
        else:
            depth = Image.open(depth_name)

        vhm = None
        if len(self.frame.columns) > 2:
            vhm_name = self.frame.loc[idx, 2] if len(self.frame.columns) > 2 else None
            _, vhm_extension = os.path.splitext(vhm_name)

            if vhm_extension == '.npy':
                vhm = np.load(vhm_name)
                vhm = vhm.reshape(vhm.shape[0], vhm.shape[1], 1)
            else:
                vhm = Image.open(vhm_name)

        sample = {'image': image, 'depth': depth, 'vhm': vhm}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.frame)


def get_training_data(batch_size=64, csv_data=''):
    '''
        Loads the training data from a CSV file

        Input parameters: batch_size, the number of samples in a batch
                          csv_file, a file containing the input data, in pairs

        Return type: a torch.utils.data.DataLoader object
    '''

    __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]}

    csv = csv_data

    # A transform is applied to the input data, converting it
    # to tensor format and normalizing it to have the same mean
    # and standard deviation of the ImageNet dataset
    #
    # This is related to the fact the SENet 'backbone' is indeed
    # trained on the ImageNet dataset, but I am not sure I fully
    # understand this.

    my_transforms = torchvision.transforms.Compose([
                                                        ToTensor(),
                                                        Normalize(__imagenet_stats['mean'],
                                                                  __imagenet_stats['std'])
                                                   ])

    transformed_training_trans = depthDataset(csv_file=csv,
                                              transform=my_transforms)

    # One should check carefully num_workers and pin_memory,
    # which could be important once we use CUDA.

    dataloader_training = DataLoader(transformed_training_trans, batch_size, num_workers=0, pin_memory=False)

    return dataloader_training


def define_model(is_resnet, is_densenet, is_senet):
    '''
        Selects a model to use.

        A pretrained CNN (ResNet, DenseNet or SENet) is
        used for transfer learning, and is 'encoded' to form our final model.

        Note: we only worked with SENet so far, other models are completely
        untested.
    '''

    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])

    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def train_main(use_cuda, args):
    '''
        The main function performing the training

        Argument: use_cuda, a bool specifying whether CUDA is available
                  args, the command line arguments, as parsed by a argparse.ArgumentParser object
    '''

    if(os.path.isfile(args.csv) == False):
        print('The specified CSV file ({}) does not exist. Quitting.'.format(args.csv))
        log_file.write('The specified CSV file ({}) does not exist. Quitting.\n'.format(args.csv))
        return

    model = define_model(is_resnet=False, is_densenet=False, is_senet=True)

    # We count the number of trainable parameters in the model and we print it

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of trainable parameters in the current model: {}'.format(pytorch_total_params))

    # Input images can be loaded in batches, resulting in a tensor of shape
    # [ nr_batches, nr_channels, x_dim, y_dim ]
    #
    # Batches do not make a big difference when running on CPU, but they
    # speed up the process a lot when running on GPU/CUDA.

    batch_size = args.batch_size
    vegetation_threshold = args.vmask

    if args.start_epoch != 0:

        if use_cuda == True:
            model = torch.nn.DataParallel(model, device_ids=[0, 1]).cuda()
            model = model.cuda()
        else:
            model = torch.nn.DataParallel(model, device_ids=[0, 1])

        state_dict = torch.load(args.model)['state_dict']
        model.load_state_dict(state_dict)

    else:
        if use_cuda == True:
            model = model.cuda()

    # Enables the cuDNN autotuner
    # Selects the Adam optimizer, with parameters as specified on the command line

    cudnn.benchmark = True
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # We get a list of pairs features/labels to be used in the training

    train_loader = get_training_data(batch_size, args.csv)

    # Optionally, we also load the test dataset

    if args.test != None:
        test_loader = get_training_data(batch_size, args.test)
    else:
        test_loader = None

    # We perform the actual training, for each epoch we calculate
    # the current adjusted learning rate, then we perform the actual
    # training, finally we save the current weights.

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        if(args.debug == False):
            train(train_loader, model, optimizer, epoch, use_cuda, vegetation_threshold)

        # If a test set has been provided, we evaluate the loss there every epoch

        if args.test != None:
            loss_on_test_set(test_loader, model, epoch, use_cuda, vegetation_threshold)

        out_name = save_model + str(epoch) + '.pth.tar'
        modelname = save_checkpoint({'state_dict': model.state_dict()}, out_name)
        print('Snapshot saved to: {}'.format(modelname))


def train(train_loader, model, optimizer, epoch, use_cuda, vegetation_threshold):
    '''
        Performs an epoch of training.

        Arguments: train_loader, a torch.utils.data.DataLoader object helping us loading the training pairs
                   model, an object containing our model
                   optimizer, our optimizer, a torch.optim.Optimizer object
                   epoch, an integer, the current epoch
                   use_cuda, a bool specifying whether we are using CUDA or not
    '''

    batch_time = AverageMeter()
    losses = AverageMeter()

    # The model is set to training mode

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)

    if use_cuda ==  True:
        get_gradient = sobel.Sobel().cuda()
    else:
        get_gradient = sobel.Sobel()

    end = time.time()
    for i, sample_batched in enumerate(train_loader):

        image, depth, vhm = sample_batched['image'], sample_batched['depth'], sample_batched['vhm']

        # Select the relevant channels
        image = image[:, (0, 1, 2, 3, 4, 5, 6, 7, 8), :, :]

        # Not sure if this resizing should go here, but it does the trick!
        image = torch.nn.functional.interpolate(image, size=(500,500), mode='bilinear')
        depth = torch.nn.functional.interpolate(depth, size=(250,250), mode='bilinear')
        vhm = torch.nn.functional.interpolate(vhm, size=(250,250), mode='bilinear')

        # We convert the VHM to a binary map, with 0's where the vegetation exceeds the threshold
        # and 1's in the areas that we actually want to be considered when training

        vhm.apply_(lambda x: 1 if x < vegetation_threshold/50 else 0)

        if use_cuda == True:
            depth = depth.cuda(non_blocking=True)
            image = image.cuda()
            vhm = vhm.cuda(non_blocking=True)

        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)
        vhm = torch.autograd.Variable(vhm)

        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float()

        if use_cuda == True:
            ones = ones.cuda()

        ones = torch.autograd.Variable(ones)

        # We initialize the gradients to zero

        optimizer.zero_grad()

        # The model is evaluated on the current feature sample

        output = model(image)

        # We apply the vegetation mask to the input and the output images
        # Note that torch.mul() performs element-wise multiplication
        
        depth = torch.mul(depth, vhm)
        output = torch.mul(output, vhm)

        # We calculate the loss function

        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        loss = loss_depth + loss_normal + (loss_dx + loss_dy)
        losses.update(loss.data, image.size(0))

        # Then we backpropagate to calculate the gradients, and we run one optimizer step

        loss.backward()
        optimizer.step()

        # Bookkeeping: calculating elapsed time, printing some useful info

        batch_time.update(time.time() - end)
        end = time.time()

        batchSize = depth.size(0)

        message = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f} ({batch_time.sum:.3f})\t' \
                  'Loss {loss.val:.4f} ({loss.avg:.4f})' \
                  .format(epoch, i, len(train_loader), batch_time=batch_time, loss=losses)

        print(message)
        log_file.write(message + '\n')


def loss_on_test_set(test_loader, model, epoch, use_cuda, vegetation_threshold):
    '''
        Given a (trained or partially trained) model, evaluates the loss on the training set.

        Arguments: test_loader, a torch.utils.data.DataLoader object helping us to load the test pairs
                   model, an object containing our model
                   epoch, the current epoch
                   use_cuda, a bool specifying whether we are using CUDA or not
    '''

    # The model is set to training mode

    model.train()

    cos = nn.CosineSimilarity(dim=1, eps=0)

    if use_cuda == True:
        get_gradient = sobel.Sobel().cuda()
    else:
        get_gradient = sobel.Sobel()

    all_losses = []
    all_maes = []
    all_mses = []

    for i, sample_batched in enumerate(test_loader):

        image, depth, vhm = sample_batched['image'], sample_batched['depth'], sample_batched['vhm']

        image = image[:, (0, 1, 2, 3, 4, 5, 6, 7, 8), :, :]

        # Not sure if this resizing should go here, but it does the trick!
        image = torch.nn.functional.interpolate(image, size=(500,500), mode='bilinear')
        depth = torch.nn.functional.interpolate(depth, size=(250, 250), mode='bilinear')
        vhm = torch.nn.functional.interpolate(vhm, size=(250,250), mode='bilinear')

        # We convert the VHM to a binary map, with 0's where the vegetation exceeds the threshold
        # and 1's in the areas that we actually want to be considered when training

        vhm.apply_(lambda x: 1 if x < vegetation_threshold/50 else 0)

        if use_cuda == True:
            depth = depth.cuda(non_blocking=True)
            image = image.cuda()
            vhm = vhm.cuda(non_blocking=True)

        image = torch.autograd.Variable(image)
        depth = torch.autograd.Variable(depth)
        vhm = torch.autograd.Variable(vhm)

        ones = torch.ones(depth.size(0), 1, depth.size(2), depth.size(3)).float()

        if use_cuda == True:
            ones = ones.cuda()

        ones = torch.autograd.Variable(ones)

        # The model is evaluated on the current feature sample

        output = model(image)

        # We apply the vegetation mask to the input and the output images
        # Note that torch.mul() performs element-wise multiplication

        depth = torch.mul(depth, vhm)
        output = torch.mul(output, vhm)

        # We calculate the loss function

        depth_grad = get_gradient(depth)
        output_grad = get_gradient(output)
        depth_grad_dx = depth_grad[:, 0, :, :].contiguous().view_as(depth)
        depth_grad_dy = depth_grad[:, 1, :, :].contiguous().view_as(depth)
        output_grad_dx = output_grad[:, 0, :, :].contiguous().view_as(depth)
        output_grad_dy = output_grad[:, 1, :, :].contiguous().view_as(depth)
        depth_normal = torch.cat((-depth_grad_dx, -depth_grad_dy, ones), 1)
        output_normal = torch.cat((-output_grad_dx, -output_grad_dy, ones), 1)
        loss_depth = torch.log(torch.abs(output - depth) + 0.5).mean()
        loss_dx = torch.log(torch.abs(output_grad_dx - depth_grad_dx) + 0.5).mean()
        loss_dy = torch.log(torch.abs(output_grad_dy - depth_grad_dy) + 0.5).mean()
        loss_normal = torch.abs(1 - cos(output_normal, depth_normal)).mean()
        loss = loss_depth + loss_normal + (loss_dx + loss_dy)

        mae = 50 * torch.mean(torch.abs(output - depth))
        mse = 50 * torch.sqrt(torch.mean((output - depth) ** 2))

        all_losses.append(loss.data.cpu().item())
        all_maes.append(mae.data.cpu().item())
        all_mses.append(mse.data.cpu().item())


    average = sum(all_losses) / len(all_losses)

    print('Average loss over the test set at epoch {} is: {}'.format(epoch, average))
    log_file.write('Average loss over the test set at epoch {} is: {}\n'.format(epoch, average))

    average_mae = sum(all_maes) / len(all_maes)

    print('Average MAE over the test set at epoch {} is: {}'.format(epoch, average_mae))
    log_file.write('Average MAE over the test set at epoch {} is: {}\n'.format(epoch, average_mae))

    average_mse = sum(all_mses) / len(all_mses)

    print('Average MSE over the test set at epoch {} is: {}'.format(epoch, average_mse))
    log_file.write('Average MSE over the test set at epoch {} is: {}\n'.format(epoch, average_mse))


def adjust_learning_rate(optimizer, epoch):
    '''
        Adjusting the learning ratio, as a function of the current epoch

        Arguments: optimizer, a torch.optim.Optimizer object
                   epoch, an integer specifying the current epoch
    '''

    lr = args.lr * (0.9 ** (epoch // 5))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    '''
        A simple class implementing a counter to calculate running averages.
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, filename='test.pth.tar'):
    '''
        Simply saves the current weights to disk

        Arguments: state, a dictionary obtained from model.state_dict()
                   filename, a string specifying the path where to save the checkpoint
    '''

    torch.save(state, filename)
    return filename


if __name__ == '__main__':

    # At first, we construct a command line parser...

    parser = argparse.ArgumentParser(description='')

    # Arguments concerning the training, hyperparameters, etc...

    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size (default: 1)')
    parser.add_argument('--vmask', default=5, type=float, help='vegetation mask threshold (default: 5)')

    # Arguments concerning input and ouput data location

    parser.add_argument('--prefix', default='trained_models')
    parser.add_argument('--data', default='model')
    parser.add_argument('--csv', default='')
    parser.add_argument('--test', default=None)
    parser.add_argument('--model', default='')

    # Arguments used only for debugging purposes

    parser.add_argument('--debug', action="store_true")

    # ...end we actually use it to parse the command line

    args = parser.parse_args()

    # We construct the prefix were the output files will be saved

    save_model = args.prefix + '/' + args.data + '_'
    log_file_path = args.prefix + '/' + args.data + '.log'

    if not os.path.exists(args.prefix):
        os.makedirs(args.prefix)

    log_file = open(log_file_path, "w")

    # Finally, we are ready to perform the training

    train_main(torch.cuda.is_available(), args)

    # Final cleanup

    log_file.close()
    print('Log written to: {}'.format(log_file_path))
