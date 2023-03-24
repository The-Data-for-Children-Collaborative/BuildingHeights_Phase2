from torch.utils.data import Dataset
import torchvision
import pandas as pd
from PIL import Image


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


class myDataset(Dataset):

    '''
        Our loader
    '''

    def __init__(self, csv_file):
        self.frame = pd.read_csv(csv_file, header=None)

    def __getitem__(self, idx):
        rgb_name = self.frame.loc[idx, 0]
        vhm_name = self.frame.loc[idx, 1]
        
        rgb = Image.open(rgb_name)
        vhm = Image.open(vhm_name)

        rgb = torchvision.transforms.functional.pil_to_tensor(rgb)
        rgb = rgb.float().div(256)

        # This keeps only the first three channels (RGB),
        # discarding the alpha channel... It works for TIFs.
        rgb = rgb[:3]

        vhm = torchvision.transforms.functional.pil_to_tensor(vhm)

        sample = {'rgb': rgb, 'vhm': vhm}

        return sample

    def __len__(self):
        return len(self.frame)
