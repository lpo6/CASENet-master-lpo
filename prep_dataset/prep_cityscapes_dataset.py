import os
import torch
import torchvision.transforms as standard_transforms
from dataloader.cityscapes_data import CityscapesData

def get_dataloader(args):
    input_size = (384, 384)
    img_transform = standard_transforms.Compose([standard_transforms.ToTensor()])
    label_transform = standard_transforms.Compose([])

    # 使用正斜杠避免转义问题
    data_root = "D:/marterial/daerchuyanfwk_detection/TRY-2/CASENet-master/cityscapes-preprocess/data_proc"

    train_dataset = CityscapesData(
        data_root + "/leftImg8bit",
        data_root + "/gtFine", 
        data_root + "/train.txt",
        data_root + "/train_label_binary_np.h5",
        input_size, args.cls_num, img_transform, label_transform
    )

    val_dataset = CityscapesData(
        data_root + "/leftImg8bit",
        data_root + "/gtFine",
        data_root + "/val.txt", 
        data_root + "/val_label_binary_np.h5",
        input_size, args.cls_num, img_transform, label_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True  # 减少workers避免问题
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=True
    )

    return train_loader, val_loader
