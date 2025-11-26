# import torch
# import torch.utils.data as data
# import torchvision.transforms as transforms
#
# import os
# import numpy as np
# import string
# import PIL
# from PIL import Image
# import time
# import zipfile
# import shutil
# import pdb
# import h5py
# import random
# import matplotlib.pyplot as plt
#
# class CityscapesData(data.Dataset):
#
#     def __init__(self, img_folder, label_folder, anno_txt, hdf5_file_name, input_size, cls_num, img_transform, label_transform):
#
#         self.img_folder = img_folder
#         self.label_folder = label_folder
#         self.input_size = input_size
#         self.cls_num = cls_num
#         self.img_transform = img_transform
#         self.label_transform = label_transform
#
#         self.h5_f = h5py.File(hdf5_file_name, 'r')
#
#         # Convert txt file to dict so that can use index to get filename.
#         cnt = 0
#         self.idx2name_dict = {}
#         self.ids = []
#         print(f"正在打开文件: {anno_txt}")
#         print(f"文件是否存在: {os.path.exists(anno_txt)}")
#         f = open(anno_txt, 'r', encoding='utf-8')
#         lines = f.readlines()
#         for line in lines:
#             row_data = line.split()
#             img_name = row_data[0]
#             label_name = row_data[1]
#             self.idx2name_dict[cnt] = {}
#             self.idx2name_dict[cnt]['img'] = img_name
#             self.idx2name_dict[cnt]['label'] = label_name
#             self.ids.append(cnt)
#             cnt += 1
#
#     def __getitem__(self, index):
#         img_name = self.idx2name_dict[index]['img']
#         label_name = self.idx2name_dict[index]['label']
#         img_path = os.path.join(self.img_folder, img_name)
#
#         # Set the same random seed for img and label transform
#         seed = np.random.randint(2147483647)
#
#         # Load img into tensor
#         img = Image.open(img_path).convert('RGB') # W X H
#
#         random.seed(seed)
#         processed_img = self.img_transform(img) # 3 X H X W
#
#         np_data = self.h5_f['data/'+label_name.replace('/', '_').replace('bin', 'npy')]
#
#         label_data = []
#         num_cls = np_data.shape[2]
#         for k in range(num_cls):
#             if np_data[:,:,num_cls-1-k].sum() > 0: # The order is reversed to be consistent with class name idx in official.
#                 random.seed(seed) # Before transform, set random seed same as img transform, to keep consistent!
#                 label_tensor = self.label_transform(torch.from_numpy(np_data[:, :, num_cls-1-k]).unsqueeze(0).float())
#             else: # ALL zeros, don't need transform, maybe a bit faster?..
#                 label_tensor = torch.zeros(1, self.input_size, self.input_size).float()
#             label_data.append(label_tensor.squeeze(0).long())
#         label_data = torch.stack(label_data).transpose(0,1).transpose(1,2) # N X H X W -> H X W X N
#         print("label_data.sum():{0}".format(label_data.sum()))
#         print("label_data.max():{0}".format(label_data.max()))
#         return processed_img, label_data
#         # processed_img: 3 X 472(H) X 472(W)
#         # label tensor: 472(H) X 472(W) X 19
#
#     def __len__(self):
#         return len(self.ids)
#

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import os
import numpy as np
import string
import PIL
from PIL import Image
import time
import zipfile
import shutil
import pdb
import h5py
import random
import matplotlib.pyplot as plt


class CityscapesData(data.Dataset):

    def __init__(self, img_folder, label_folder, anno_txt, hdf5_file_name, input_size, cls_num, img_transform,
                 label_transform):

        self.img_folder = img_folder
        self.label_folder = label_folder
        self.input_size = input_size
        self.cls_num = cls_num
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.hdf5_file_name = hdf5_file_name  # 保存文件路径而不是文件对象

        # Convert txt file to dict so that can use index to get filename.
        cnt = 0
        self.idx2name_dict = {}
        self.ids = []
        print(f"正在打开文件: {anno_txt}")
        print(f"文件是否存在: {os.path.exists(anno_txt)}")
        f = open(anno_txt, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            row_data = line.split()
            img_name = row_data[0]
            label_name = row_data[1]
            self.idx2name_dict[cnt] = {}
            self.idx2name_dict[cnt]['img'] = img_name
            self.idx2name_dict[cnt]['label'] = label_name
            self.ids.append(cnt)
            cnt += 1
        f.close()

    # def __getitem__(self, index):
    #     img_name = self.idx2name_dict[index]['img']
    #     label_name = self.idx2name_dict[index]['label']
    #
    #     if img_name.startswith('/'):
    #         img_name = img_name[1:]
    #
    #     img_path = os.path.join(self.img_folder, img_name)
    #     # Set the same random seed for img and label transform
    #     seed = np.random.randint(2147483647)
    #
    #     # Load img into tensor
    #     img = Image.open(img_path).convert('RGB')  # W X H
    #
    #     random.seed(seed)
    #     processed_img = self.img_transform(img)  # 3 X H X W
    #
    #     # 在__getitem__中打开h5文件，确保每个进程有独立的文件句柄
    #     with h5py.File(self.hdf5_file_name, 'r') as h5_f:
    #         # 方法：取文件名部分，去掉路径和扩展名
    #         label_basename = os.path.basename(label_name)  # zurich_000116_000019_gtFine_edge.bin
    #         label_key = label_basename.replace('.bin', '')  # zurich_000116_000019_gtFine_edge
    #
    #         np_data = h5_f[label_key][:]
    #
    #     label_data = []
    #     num_cls = np_data.shape[2]
    #     for k in range(num_cls):
    #         if np_data[:, :,
    #            num_cls - 1 - k].sum() > 0:  # The order is reversed to be consistent with class name idx in official.
    #             random.seed(seed)  # Before transform, set random seed same as img transform, to keep consistent!
    #             label_tensor = self.label_transform(
    #                 torch.from_numpy(np_data[:, :, num_cls - 1 - k]).unsqueeze(0).float())
    #         else:  # ALL zeros, don't need transform, maybe a bit faster?..
    #             label_tensor = torch.zeros(1, self.input_size, self.input_size).float()
    #         label_data.append(label_tensor.squeeze(0).long())
    #     label_data = torch.stack(label_data).transpose(0, 1).transpose(1, 2)  # N X H X W -> H X W X N
    #     print("label_data.sum():{0}".format(label_data.sum()))
    #     print("label_data.max():{0}".format(label_data.max()))
    #     return processed_img, label_data
    #     # processed_img: 3 X 472(H) X 472(W)
    #     # label tensor: 472(H) X 472(W) X 19
    def __getitem__(self, index):
        img_name = self.idx2name_dict[index]['img']
        label_name = self.idx2name_dict[index]['label']

        if img_name.startswith('/'):
            img_name = img_name[1:]

        img_path = os.path.join(self.img_folder, img_name)
        seed = np.random.randint(2147483647)

        # Load img into tensor
        img = Image.open(img_path).convert('RGB')

        random.seed(seed)
        processed_img = self.img_transform(img)

        # 🚨 激进优化：分批处理类别，避免同时处理所有类别
        with h5py.File(self.hdf5_file_name, 'r') as h5_f:
            label_basename = os.path.basename(label_name)
            label_key = label_basename.replace('.bin', '')
            np_data_2d = h5_f[label_key][:]  # (1024, 2048)

        h, w = np_data_2d.shape
        num_classes = self.cls_num

        # 🚨 分批处理：每次只处理几个类别
        batch_size = 5  # 每次处理5个类别
        all_label_tensors = []

        for batch_start in range(0, num_classes, batch_size):
            batch_end = min(batch_start + batch_size, num_classes)
            batch_label_data = []

            for cls_idx in range(batch_start, batch_end):
                bit_mask = 1 << cls_idx
                cls_mask = torch.from_numpy((np_data_2d & bit_mask).astype(bool)).float().unsqueeze(0)

                random.seed(seed)
                label_tensor = self.label_transform(cls_mask)
                batch_label_data.append(label_tensor.squeeze(0).long())

                del cls_mask, label_tensor

            # 处理完一个批次后立即清理
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            all_label_tensors.extend(batch_label_data)
            del batch_label_data

        label_data = torch.stack(all_label_tensors).transpose(0, 1).transpose(1, 2)  # H X W X N

        # 保留调试打印
        print("label_data.sum():{0}".format(label_data.sum()))
        print("label_data.max():{0}".format(label_data.max()))

        # 🚨 最终清理
        del np_data_2d, all_label_tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return processed_img, label_data
        # processed_img: 3 X 384(H) X 384(W)
        # label tensor: 384(H) X 384(W) X 19
    def __len__(self):
        return len(self.ids)