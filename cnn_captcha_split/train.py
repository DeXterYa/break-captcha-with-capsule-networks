from model_architectures import ConvolutionalNetwork
# from model_architectures_mini import ConvolutionalNetwork
from data_providers import CaptchaDigit90
from experiment_builder import ExperimentBuilder
import os
from arg_extractor import get_args
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split

args, device = get_args()  # get arguments from command line
rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
torch.manual_seed(seed=args.seed)  # sets pytorch's seed

if args.device == "cuda":
    use_gpu = True
else:
    use_gpu = False

# The size of captcha images
height = 28
width = 54
# TODO: add mean and std in the future
image_shape = (3, 28, 28)

transform = transforms.Compose([
    transforms.ToTensor()
])

coord = False
if args.coord == 'True':
    coord = True
    print('Add coordinates to the feature maps')

random_order = True

if args.random_order == 'False':
    random_order = False
    print('Input images are in correct order')
else:
    print('Input images are in random order')

random_rotation = True
if args.random_rotation == 'False':
    random_rotation = False
    print('Input images are not rotated')
else:
    print('Input images are randomly rotated')

# Split the images in /train folder into training set and validation set
print("The total number of data used in training and validation: ", args.num_train_val)
num_train_val = args.num_train_val
num_test = 10000


# train_val_filenames = [str(i).zfill(filename_length) for i in range(num_train_val)]
# test_filenames = [str(i).zfill(filename_length) for i in range(num_test)]
# # Split into training set and validation set
# train_filenames, val_filenames = train_test_split(train_val_filenames,
#                                                   test_size=10000, random_state=args.seed)  # fixed split with the seed

tran_val_idx = [i for i in range(num_train_val)]
train_idx, val_idx = train_test_split(tran_val_idx, test_size=num_test, random_state=args.seed)

train_set = CaptchaDigit90(root_dir=args.data_path, set_name="train", index=train_idx)
val_set = CaptchaDigit90(root_dir=args.data_path, set_name="val", index=val_idx)
test_set = CaptchaDigit90(root_dir=args.data_path, set_name="test")

train_data = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
val_data = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=1)
test_data = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)

network = ConvolutionalNetwork(input_shape=(args.batch_size, *image_shape), num_char=3, num_dim=10,
                               coord=coord, random_order=random_order, random_rotation=random_rotation)

exp = ExperimentBuilder(network_model=network, use_gpu=use_gpu, experiment_name=args.experiment_name,
                        num_epochs=args.epochs, weight_decay_coefficient=args.weight_decay_coefficient,
                        continue_from_epoch=args.continue_from_epoch, train_data=train_data,
                        val_data=val_data, test_data=test_data)

experiment_metrics, test_metrics = exp.run_experiment()
