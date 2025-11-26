import os
import torch
import torchvision.transforms as standard_transforms

from dataloader.cityscapes_data import CityscapesData

train_adress = 'train_label_binary_np.h5'
val_adress = 'val_label_binary_np.h5'

def get_dataloader(args):
    # Setup data transforms
    input_size = (192,384)

    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
    ])

    label_transform = standard_transforms.Compose([
    ])

    # Setup data paths - 使用原始字符串避免转义问题
    data_root = r"D:\marterial\daerchuyanfwk_detection\TRY-2\CASENet-master\cityscapes-preprocess\data_proc"

    root_img_folder = data_root
    root_label_folder = data_root
    train_anno_txt = os.path.join(data_root, 'train.txt')
    val_anno_txt = os.path.join(data_root, 'val.txt')
    train_hdf5_file = os.path.join(data_root, train_adress)
    val_hdf5_file = os.path.join(data_root, val_adress)

    # Create datasets
    train_dataset = CityscapesData(
        root_img_folder, root_label_folder, train_anno_txt, train_hdf5_file,
        input_size, args.cls_num, img_transform, label_transform
    )

    val_dataset = CityscapesData(
        root_img_folder, root_label_folder, val_anno_txt, val_hdf5_file,
        input_size, args.cls_num, img_transform, label_transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )

    return train_loader, val_loader
