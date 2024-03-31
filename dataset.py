import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import ImageTransforms


class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, stage, crop_size, scaling_factor, lr_img_type, hr_img_type):
        """
        :param data_folder: # folder with JSON data files
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """

        self.data_folder = data_folder
        self.stage = stage.lower()
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type

        assert self.stage in {'train', 'test'}

        assert lr_img_type in {'[0, 255]',
                               '[0, 1]', '[-1, 1]', 'imagenet-norm'}
        assert hr_img_type in {'[0, 255]',
                               '[0, 1]', '[-1, 1]', 'imagenet-norm'}

        # If this is a training dataset, then crop dimensions must be perfectly divisible by the scaling factor
        # (If this is a test dataset, images are not cropped to a fixed size, so this variable isn't used)
        if self.stage == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop dimensions are not perfectly divisible by scaling factor! This will lead to a mismatch in the dimensions of the original HR patches and their super-resolved (SR) versions!"

        # Read list of image-paths
        if self.stage == 'train':
            with open(os.path.join(data_folder, 'train_images.json'), 'r') as j:
                self.images = json.load(j)
        else:
            with open(os.path.join(data_folder, 'test_images.json'), 'r') as j:
                self.images = json.load(j)

        # Select the correct set of transforms
        self.transform = ImageTransforms(split=self.stage,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor,
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type)

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img = Image.open(os.path.join(
            self.data_folder, self.images[i]), mode='r')
        img = img.convert('RGB')
        if img.width <= self.crop_size or img.height <= self.crop_size:
            print(self.images[i], img.width, img.height)
            del self.images[i]
            return self.__getitem__(i)
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.images)


def prepare_data():
    train_paths = []
    test_paths = []
    data_folder = 'data'
    folders = sorted([x for x in os.listdir(os.path.join(
        os.getcwd(), data_folder)) if os.path.isdir(os.path.join(data_folder, x))])
    for folder in folders:
        classes = sorted([x for x in os.listdir(
            os.path.join(data_folder, folder))])
        for class_name in classes:
            image_paths = sorted([os.path.join(folder, class_name, x) for x in os.listdir(
                os.path.join(data_folder, folder, class_name)) if x.endswith(('.jpg', '.png', 'JPEG'))])
            image_paths = [x for x in image_paths if all(
                x > 96 for x in Image.open(os.path.join(data_folder, x)).size)]
            if 'train' in folder:
                train_paths.extend(image_paths)
            else:
                test_paths.extend(image_paths)

    print(len(train_paths), len(test_paths))
    with open(os.path.join(data_folder, 'train_images.json'), 'w') as j:
        json.dump(train_paths, j)
    with open(os.path.join(data_folder, 'test_images.json'), 'w') as j:
        json.dump(test_paths, j)


if __name__ == '__main__':
    prepare_data()
