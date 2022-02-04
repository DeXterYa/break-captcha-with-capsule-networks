import argparse
import torch
def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--continue_from_epoch', nargs="?", type=int, default=-1, help='Starting epochs')
    parser.add_argument('-e', '--epochs', type=int, default=50, help='Number of epochs.')
    parser.add_argument('--experiment_name', nargs="?", type=str, default="exp_1",
                        help='Experiment name - to be used for building the experiment folder')
    # Data directory
    parser.add_argument('--data_path', type=str, default="data/CAPTCHA_3digits_noise",
                        help='Path to the MNIST or CIFAR dataset. Alternatively you can set the path as an environmental variable $data.')
    parser.add_argument('--dataset', nargs='?', type=str, default='mnist', help="Choose the dataset: 'mnist' or 'cifar10'.")
    # Select device "cuda" for GPU or "cpu"
    parser.add_argument('--device', type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
                        choices=['cuda', 'cpu'], help='Device to use. Choose "cuda" for GPU or "cpu".')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--lr_decay', type=float, default=0.96, help='Exponential learning rate decay.')
    parser.add_argument('--num_routing', type=int, default=3, help='Number of routing iteration in routing capsules.')
    parser.add_argument('--seed', nargs="?", type=int, default=7112018,
                        help='Seed to use for random number generator for experiment')
    parser.add_argument('--weight_decay_coefficient', nargs="?", type=float, default=1e-05,
                        help='Weight decay to use for Adam')
    parser.add_argument('--num_primary_channel', type=int, default=256)
    parser.add_argument('--dropout', type=str, default='False')
    parser.add_argument('--test_name', type=str, default='test')
    parser.add_argument('--num_train_val', type=int, default=60000)
    parser.add_argument('--coord', type=str, default='False')
    parser.add_argument('--attention', type=str, default='False')
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--random_order', nargs='?', type=str, default='True')
    parser.add_argument('--random_rotation', nargs='?', type=str, default='True')
    args = parser.parse_args()
    device = torch.device(args.device)

    return args, device
