# import numpy as np
# from PIL import Image
# import os
# import zipfile
# import shutil
# import h5py
# from tqdm import tqdm
#
# import torch
#
# def convert_num_to_bitfield(label_data, h, w, npz_name, root_folder, h5_file, cls_num=19):
#     label_list = list(label_data)
#     all_bit_tensor_list = []
#     for n in label_list: # Iterate in each pixel
#         # Convert a value to binary format in each bit.
#         bitfield = np.asarray([int(digit) for digit in bin(n)[2:]])
#         bit_tensor = torch.from_numpy(bitfield)
#         actual_len = bit_tensor.size()[0]
#         padded_bit_tensor = torch.cat((torch.zeros(cls_num-actual_len).byte(), bit_tensor.byte()), dim=0)
#         all_bit_tensor_list.append(padded_bit_tensor)
#     all_bit_tensor_list = torch.stack(all_bit_tensor_list).view(h, w, cls_num)
#     h5_file.create_dataset('data/'+npz_name.replace('/', '_'), data=all_bit_tensor_list.numpy())
#
# if __name__ == "__main__":
#     f = open(r"D:\marterial\daerchuyanfwk_detection\TRY-2\CASENet-master/cityscapes-preprocess/data_proc/val.txt", 'r')
#     lines = f.readlines()
#     root_folder = r"D:\marterial\daerchuyanfwk_detection\TRY-2\CASENet-master/cityscapes-preprocess/data_proc/"
#
#     h5_file = h5py.File("val_label_binary_np.h5", 'w')
#     for ori_line in tqdm(lines):
#         line = ori_line.split()
#         bin_name = line[1]
#         img_name = line[0]
#
#         label_path = os.path.join(root_folder, bin_name)
#         img_path = os.path.join(root_folder, img_name)
#
#         img = Image.open(img_path).convert('RGB')
#         w, h = img.size # Notice: not h, w! This is very important! Otherwise, the label is wrong for each pixel.
#
#         label_data = np.fromfile(label_path, dtype=np.uint32)
#         npz_name = bin_name.replace("bin", "npy")
#         convert_num_to_bitfield(label_data, h, w, npz_name, root_folder, h5_file)

import os
import h5py
import numpy as np
from tqdm import tqdm
import time


def analyze_file_structure():
    """分析实际的文件结构"""
    data_proc = r'D:\marterial\daerchuyanfwk_detection\TRY-2\CASENet-master\cityscapes-preprocess\data_proc_small'

    print("🔍 分析文件结构...")

    # 检查gtFine目录的实际结构
    gtfine_dir = os.path.join(data_proc, 'gtFine')
    if os.path.exists(gtfine_dir):
        for split in ['train', 'val']:
            split_dir = os.path.join(gtfine_dir, split)
            if os.path.exists(split_dir):
                cities = os.listdir(split_dir)
                print(fr"\n📁 {split}集 - 城市数量: {len(cities)}")
                for city in cities[:3]:  # 显示前3个城市
                    city_dir = os.path.join(split_dir, city)
                    bin_files = [f for f in os.listdir(city_dir) if f.endswith('.bin')]
                    if bin_files:
                        print(f"   {city}: {len(bin_files)}个bin文件")
                        print(f"     示例: {bin_files[0]}")


def convert_dataset_working(data_proc, dataset_type):
    """
    真正可工作的HDF5转换版本
    """
    print(fr"\n🎯 开始处理 {dataset_type} 集...")

    list_file = os.path.join(data_proc, f'{dataset_type}.txt')
    hdf5_path = os.path.join(data_proc, f'{dataset_type}_label_binary_np_small.h5')

    if not os.path.exists(list_file):
        print(f"❌ 文件列表不存在: {list_file}")
        return 0

    with open(list_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"📊 总数据行数: {len(lines)}")

    success_count = 0
    start_time = time.time()

    # 删除已存在的HDF5文件
    if os.path.exists(hdf5_path):
        os.remove(hdf5_path)

    with h5py.File(hdf5_path, 'w') as hf:
        for i, line in enumerate(tqdm(lines, desc=f"转换{dataset_type}")):
            try:
                parts = line.split()
                if not parts:
                    continue

                image_path = parts[0]

                if dataset_type == 'test':
                    # 测试集 - 创建空的占位符数据
                    placeholder = np.zeros((512,1024), dtype=np.uint32)
                    # 使用简单键名
                    dataset_key = f"image_{i:06d}"
                    hf.create_dataset(dataset_key, data=placeholder, compression='gzip')
                    success_count += 1
                else:
                    # 训练集/验证集 - 使用真实bin数据
                    # 解析路径: /leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png
                    path_parts = image_path.split('/')
                    if len(path_parts) >= 4:
                        city_name = path_parts[3]  # aachen
                        file_base = path_parts[4].replace('_leftImg8bit.png', '')

                        # 构建正确的bin文件路径
                        bin_path = os.path.join(
                            data_proc,
                            'gtFine',
                            dataset_type,
                            city_name,
                            f'{file_base}_gtFine_edge.bin'
                        )

                        if os.path.exists(bin_path):
                            # 读取bin文件
                            with open(bin_path, 'rb') as bin_file:
                                binary_data = np.fromfile(bin_file, dtype=np.uint32)

                            # 重塑为图像尺寸 (524288)
                            if binary_data.size == 524288:
                                image_data = binary_data.reshape(512,1024)
                            else:
                                # 计算合适的尺寸
                                height = 512
                                width = binary_data.size // height
                                image_data = binary_data.reshape(height, width)

                            # 使用简单键名保存到HDF5
                            dataset_key = f"image_{i:06d}"
                            hf.create_dataset(dataset_key, data=image_data, compression='gzip')
                            success_count += 1

                            # 显示前几个成功的文件
                            if success_count <= 3:
                                print(f"   ✅ 成功转换: {os.path.basename(bin_path)} -> {dataset_key}")
                        else:
                            if i < 5:  # 只显示前几个错误
                                print(f"   ❌ Bin文件不存在: {bin_path}")
                    else:
                        if i < 5:
                            print(f"   ❌ 路径格式错误: {image_path}")

            except Exception as e:
                print(f"❌ 处理失败 (行 {i}): {e}")
                continue

    total_time = (time.time() - start_time) / 60
    file_size = os.path.getsize(hdf5_path) / (1024 * 1024) if os.path.exists(hdf5_path) else 0

    print(f"✅ {dataset_type}集完成: {success_count}/{len(lines)} 文件")
    print(f"📁 输出大小: {file_size:.1f} MB")

    return success_count


def create_simple_solution():
    """
    最简单的解决方案：直接遍历gtFine目录
    """
    print(r"\n🔄 尝试简单解决方案...")

    data_proc = r'D:\marterial\daerchuyanfwk_detection\TRY-2\CASENet-master\cityscapes-preprocess\data_proc_small'

    for dataset_type in ['train', 'val']:
        print(fr"\n🎯 处理 {dataset_type} 集...")

        hdf5_path = os.path.join(data_proc, f'{dataset_type}_label_binary_np_small.h5')
        gtfine_dir = os.path.join(data_proc, 'gtFine', dataset_type)

        if not os.path.exists(gtfine_dir):
            print(f"❌ 目录不存在: {gtfine_dir}")
            continue

        # 删除已存在的HDF5文件
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)

        success_count = 0
        with h5py.File(hdf5_path, 'w') as hf:
            # 遍历所有城市目录
            for city in os.listdir(gtfine_dir):
                city_dir = os.path.join(gtfine_dir, city)
                if os.path.isdir(city_dir):
                    # 处理所有bin文件
                    bin_files = [f for f in os.listdir(city_dir) if f.endswith('_edge.bin')]

                    for bin_file in tqdm(bin_files, desc=f"处理{city}"):
                        try:
                            bin_path = os.path.join(city_dir, bin_file)

                            # 读取bin文件
                            with open(bin_path, 'rb') as f:
                                binary_data = np.fromfile(f, dtype=np.uint32)

                            # 重塑为512x1024
                            if binary_data.size == 524288:
                                image_data = binary_data.reshape(512,1024)
                            else:
                                image_data = binary_data.reshape(512,1024)  # 强制重塑

                            # 保存到HDF5，使用原文件名作为键
                            dataset_key = bin_file.replace('.bin', '')
                            hf.create_dataset(dataset_key, data=image_data, compression='gzip')
                            success_count += 1

                        except Exception as e:
                            print(f"❌ 处理失败 {bin_file}: {e}")
                            continue

        file_size = os.path.getsize(hdf5_path) / (1024 * 1024) if os.path.exists(hdf5_path) else 0
        print(f"✅ {dataset_type}集完成: {success_count} 个文件")
        print(f"📁 输出大小: {file_size:.1f} MB")


def verify_final_results():
    """验证最终结果"""
    print(r"\n🔍 验证最终HDF5文件...")

    data_proc = r'D:\marterial\daerchuyanfwk_detection\TRY-2\CASENet-master\cityscapes-preprocess\data_proc_small'

    for dataset in ['train', 'val', 'test']:
        hdf5_path = os.path.join(data_proc, f'{dataset}_label_binary_np_small.h5')

        if os.path.exists(hdf5_path):
            size = os.path.getsize(hdf5_path) / (1024 * 1024)
            print(fr"\n📊 {dataset}: {size:.1f} MB")

            try:
                with h5py.File(hdf5_path, 'r') as hf:
                    keys = list(hf.keys())
                    print(f"   数据集数量: {len(keys)}")

                    if keys:
                        sample_key = keys[0]
                        sample_data = hf[sample_key][:]
                        print(f"   样本形状: {sample_data.shape}")
                        print(f"   数据类型: {sample_data.dtype}")
                        print(f"   数据范围: {sample_data.min()} ~ {sample_data.max()}")
            except Exception as e:
                print(f"   ❌ 验证失败: {e}")
        else:
            print(f"❌ {dataset}: 文件不存在")


def main():
    print("🚀 启动完全修复的HDF5转换")

    # 先分析结构
    analyze_file_structure()

    # 使用简单解决方案（推荐）
    create_simple_solution()

    # 验证结果
    verify_final_results()

    print(r"\n🎉 所有转换完成！")


if __name__ == '__main__':
    main()