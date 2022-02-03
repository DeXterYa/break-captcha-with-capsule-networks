from image import ImageCaptcha
from tqdm import tqdm
import os
import random
import torch
import torchvision.transforms as transforms

DATASET = "CAPTCHA_3digits_noise"
num_train = 60000
num_test = 10000
num_digit = 3
partition_size = 10000

img = ImageCaptcha(width = 120, height = 72)

transform = transforms.Compose([
    transforms.ToTensor()
])

# record the frequency of each digit in train and test dataset
stats_train = [0 for x in range(10)]
stats_test = [0 for x in range(10)]


# Generate training data
for i in tqdm(range(num_train)):

    # create a directory for the training example
    # directory_name = str(i).zfill(num_length)
    # full_directory = DATASET + '/train/' + directory_name
    # os.mkdir(full_directory)
    
    # create a .pt file for every 10,000 example
    if i % partition_size == 0:
        images = []
        labels = []
    # generate the digit sequence
    label = []
    sequence = ''
    for d in range(num_digit):
        digit = random.randint(0, 9)
        stats_train[digit] += 1
        label.append(torch.tensor([digit]))
        sequence += str(digit)

    label = torch.cat(label)
    # generate captcha image and its mask with the sequence
    # image_path = full_directory + '/' + sequence + '.png'
    # image_mask_path = full_directory + '/' + sequence + '_mask.png'
    image = transform(img.generate(sequence))

    images.append(image)
    labels.append(label)

    if (i + 1) % partition_size == 0:
        save_path = DATASET + '/train/images_' + str(int((i+1) / partition_size)) +'.pt'
        torch.save({'images': images, 'labels': labels}, save_path)


# Generate test data
for i in tqdm(range(num_test)):
    if i % partition_size == 0:
        images = []
        labels = []
    
    label = []
    sequence = ''
    for d in range(num_digit):
        digit = random.randint(0, 9)
        stats_train[digit] += 1
        label.append(torch.tensor([digit]))
        sequence += str(digit)

    label = torch.cat(label)

    image = transform(img.generate(sequence))

    images.append(image)
    labels.append(label)

    if (i + 1) % partition_size == 0:
        save_path = DATASET + '/test/images.pt'
        torch.save({'images': images, 'labels': labels}, save_path)

# output stats
f = open(DATASET + '/stats.txt', "w")
# f.write("# digits in the training data:\n")
# for i, number in enumerate(stats_train):
#     f.write("The freqency of " + str(i) + ": " + str(number) + '\n')
f.write("# digits in the test data:\n")
for i, number in enumerate(stats_test):
    f.write("The freqency of " + str(i) + ": " + str(number) + '\n')
f.close()