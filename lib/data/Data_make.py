from pyts.approximation import DiscreteFourierTransform
from sklearn.preprocessing import StandardScaler
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import csv
import glob
from scipy import signal
import os.path
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.autograd import Variable
import numpy as np
from PIL import Image
import cv2


# Change the path according to your data path
# Haptic denoising and downsampling
# feel&poke = all haptic
def haptic_sub():
    for n in range(1,64):
        if n <= 9:
            for m in range(1, 6):
                for batch in ['NF', 'NP', 'R8LO', 'R8RO', 'R8RU']:
                    feel_path = 'data/object_0' + str(n) + '/observation_' + str(m) + '/haptic_poke_noise/Tactile_poke_' + batch + '.csv'
                    back_path = 'data/object_0' + str(n) + '/Background/poke/Background_poke_' + batch + '.csv'
                    out_path = 'data/outputdata/haptic_subback10000/object_' + str(n) + '_' + str(m) + '_poke' + batch + '.csv'
                    readData1 = pd.read_csv(feel_path, header=None)
                    readData2 = pd.read_csv(back_path, header=None)
                    subresult = readData1 - readData2  # denoising
                    result = signal.resample(subresult, 10000, t=None, axis=0)
                    # result = StandardScaler().fit_transform(result)
                    df = pd.DataFrame(result)
                    df.to_csv(out_path,index=False,header=None)
        if n >= 10:
            for m in range(1,6):
                for batch in ['NF', 'NP', 'R8LO', 'R8RO', 'R8RU']:
                    feel_path = 'data/object_' + str(n) + '/observation_' + str(m) + '/haptic_poke_noise/Tactile_poke_' + batch + '.csv'
                    back_path = 'data/object_' + str(n) + '/Background/poke/Background_poke_' + batch + '.csv'
                    out_path = 'data/outputdata/haptic_subback10000/object_' + str(n) + '_' + str(m) + '_poke' + batch + '.csv'
                    readData1 = pd.read_csv(feel_path, header=None)
                    readData2 = pd.read_csv(back_path, header=None)
                    result = readData1 - readData2  # denoising
                    result = signal.resample(result, 10000, t=None, axis=0)  # downsampling
                    # result = StandardScaler().fit_transform(result)
                    df = pd.DataFrame(result)
                    df.to_csv(out_path, index=False, header=None)


# Haptic concatenating
def make_seq():
    in_path = 'data/outputdata/haptic_subback10000tag/*.csv'
    csv_list = glob.glob(in_path)
    csv_train = []
    csv_test = []
    csv = []
    # add tag
    # for file in csv_list:
    #     table = str(int(file.split("_")[2]) - 1)
    #     f = open(file,'r+',encoding='utf-8')
    #     content = f.read()
    #     f.seek(0, 0)
    #     f.write(table + '\n' + content)
    #     f.close()

    # concatenate
    for file in csv_list:
        df = pd.read_csv(file, header=None)
        data = df.values
        data = list(map(list, zip(*data)))
        data = pd.DataFrame(data)
        # if len(csv_train) == 0:
        #     csv_train = data
        # else:
        #     if int(file.split("_")[3]) == 3:
        #         if len(csv_test) == 0:
        #             csv_test = data
        #         else:
        #             csv_test = pd.concat([csv_test, data], axis=0)
        #     else:
        #         csv_train = pd.concat([csv_train, data], axis=0)
        if len(csv) == 0:
            csv = data
        else:
            csv = pd.concat([csv, data], axis=0)
    # csv_train.to_csv('data/outputdata/h_train.csv', index=False, header=None)
    # csv_test.to_csv('data/outputdata/h_test.csv', index=False, header=None)
    csv.to_csv('data/outputdata/h_all.csv', index=False, header=None)


# npy2csv
def npy2csv(path, topath):
    npfile = np.load(path)
    npfile = np.array(npfile)
    npfile = np.reshape(npfile, (2520, 2048))
    np_to_csv = pd.DataFrame(data=npfile)
    np_to_csv.to_csv(topath)


# npy2csv("./tag_test.csv.npy", "tag_test.csv")
# npy2csv("./tag_train.csv.npy", "tag_train.csv")
# npy2csv("./visual2_test.csv.npy", "visual2_test.csv")
# npy2csv("./visual2_train.csv.npy", "visual2_train.csv")


# Kinesthetic concatenating
def make_k():
    in_path = 'data/outputdata/k1coltag/*.csv'
    csv_list = glob.glob(in_path)
    csv_train = []
    csv_test = []
    test = []
    train = []
    csv = []
    # add tag
    # for file in csv_list:
    #     table = str(int(file.split("_")[1]) - 1)
    #     f = open(file,'r+',encoding='utf-8')
    #     content = f.read()
    #     f.seek(0, 0)
    #     f.write(table + '\n' + content)
    #     f.close()
    #
    # concatenate
    for file in csv_list:
        df = pd.read_csv(file, header=None)
        data = df.values
        data = list(map(list, zip(*data)))
        data = pd.DataFrame(data)
        if len(csv_train) == 0:
            csv_train = data
        else:
            if int(file.split("_")[2]) == 3:
                if len(csv_test) == 0:
                    csv_test = data
                else:
                    csv_test = pd.concat([csv_test, data], axis=0)
            else:
                csv_train = pd.concat([csv_train, data], axis=0)
    test = pd.DataFrame(np.tile(csv_test, (10, 1)))
    train = pd.DataFrame(np.tile(csv_train, (10, 1)))

    train.to_csv('data/outputdata/k_train.csv', index=False, header=None)
    test.to_csv('data/outputdata/k_test.csv', index=False, header=None)
    # csv.to_csv('data/outputdata/kinesthetics.csv', index=False, header=None)


# Visual feature pre-extraction
features_dir = 'F:\\AU data\\data\\outputdata\\visual1'  # visual input path
data_test = []
data_train = []
tag_test = []
tag_train = []
path = 'F:\\AU data\\data\\outputdata\\visual1'  # visual output path
def mean():
    # Calculating the mean and variance of the visual data
    img_h, img_w = 256, 256
    means, stdevs = [], []
    img_list = []
    imgs_path_list = os.listdir(path)  # imgs
    len_ = len(imgs_path_list)
    i = 0
    for item in imgs_path_list:
        img = cv2.imread(os.path.join(path, item), -1)
        # shape = img.shape # shape (3472,4640,3)
        img = cv2.resize(img, (img_w, img_h))
        img = np.reshape(img, (256, 256, -1))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        print(i, '/', len_)
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):  # rgb
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    # BGR --> RGB
    means.reverse()
    stdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))

# normMean = [0.8224512, 0.774427, 0.6780131]
# normStd = [0.16245939, 0.1814703, 0.19750762]


# Visualisation
def transform_invert(img_, transform_train):
    """
    将data 进行反transfrom操作
    :param img_: tensor
    :param transform_train: torchvision.transforms
    :return: PIL image
    """
    norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
    mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
    std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
    img_.mul_(std[:, None, None]).add_(mean[:, None, None])
    img_ = img_.transpose(0, 2).transpose(0, 1)  # C*H*W --> H*W*C
    img_ = np.array(img_) * 255

    # RGB images
    if img_.shape[2] == 3:
        img_ = Image.fromarray(img_.astype('uint8')).convert('RGB')
        # Grayscale images
    elif img_.shape[2] == 1:
        img_ = Image.fromarray(img_.astype('uint8').squeeze())
    else:
        raise Exception("Invalid img shape, expected 1 or 3 in axis 2, but got {}!".format(img_.shape[2]))
    return img_


def extraction_v():
    for filename in os.listdir(path):
        transform1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.8224512, 0.774427, 0.6780131],
                             std=[0.16245939, 0.1814703, 0.19750762])
        ]  # to Tensor
        )
        tag = int(filename.split("_")[1]) - 1
        print(filename, '\n')
        img = Image.open(path + '/' + filename)
        img1 = transform1(img)
        img2 = transform_invert(img1, transform1)
        plt.imshow(img2)
        plt.show()
        # resnet18 = models.resnet18(pretrained = True)
        inception_v3_feature_extractor = models.inception_v3(pretrained=True)  # inception_v3
        inception_v3_feature_extractor.fc = nn.Linear(2048, 2048)
        torch.nn.init.eye(inception_v3_feature_extractor.fc.weight)

        for param in inception_v3_feature_extractor.parameters():
            param.requires_grad = False
        # resnet152 = models.resnet152(pretrained = True)
        # densenet201 = models.densenet201(pretrained = True)
        print(img1.shape)
        x = Variable(torch.unsqueeze(img1, dim=0).float(), requires_grad=False)
        # x = Variable(img1, requires_grad=False)
        print(x.shape)
        inception_v3_feature_extractor.training=False
        y = inception_v3_feature_extractor(x)
        y = y.data.numpy()
        if int((filename.split("_")[2]).split(".")[0]) <= 10:
            tag_test.append(tag)
            data_test.append(y)
        else:
            tag_train.append(tag)
            data_train.append(y)
        # data_npy = np.array(data_list)
        # print(data_npy.shape)
    # np.save('visual2_test.csv', data_test)
    # np.save('visual2_train.csv', data_train)
    # np.save('tag_test.csv', tag_test)
    # np.save('tag_train.csv', tag_train)
