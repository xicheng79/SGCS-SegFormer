# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset
import mmcv
from .spectral_aware_sampling.spectral_aware_sampling import SpectralAwareSampling
from osgeo import gdal
from mmcv.utils import print_log
from mmseg.utils import get_root_logger

@DATASETS.register_module()
class GreenlandDataset(CustomDataset):
    """PanoOptiNet dataset."""

    CLASSES = ('background', 'greenland')

    PALETTE = [[0, 0, 0], [255, 255, 255]]
    
    # 覆写父类load_annotations方法，实现PanoOptiNet数据集加载的逻辑
    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_path = img_dir +'\\'+ img_name + img_suffix
                seg_path = ann_dir +'\\'+ img_name + seg_map_suffix
                # img_path = gdal.Open(img_path)
                #预处理影像
                preseg = SpectralAwareSampling(img_path, seg_path, img_infos)
                img_infos = preseg.main()
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos
    
    def get_ann_info(self, idx):
        """通过索引获取注释信息。

        参数:
            idx (int): 数据的索引。

        返回:
            dict: 指定索引的注释信息。
        """
        
        return self.img_infos[idx]['annotation']
    
    def prepare_train_img(self, idx):
        """在数据处理管道之后获取训练数据和注释。

        参数:
            idx (int): 数据的索引。

        返回:
            dict: 经过数据处理管道后的训练数据和注释，包含管道引入的新键。
        """
        # 将self.img_infos这个数组按照次序每100个变成一个子数组，赋值给self.img_infos_good这个数组
        self.img_infos_dim2 = [self.img_infos[i:i+100] for i in range(0, len(self.img_infos), 100)]
        
        bigPicIndex = int(idx / 100)
        smallPicIndex = idx % 100
        
        img_info = None
        
        for info in self.img_infos_dim2[bigPicIndex]:
            filename = info['filename']
            
            # 获取文件名中_分割的倒数第三个数字
            file_idx = int(filename.split('/')[-1].split('.')[0].split('_')[-3])
            if (file_idx - 1) == smallPicIndex:
                img_info = info
                break
        # print(img_info)
        ann_info = self.get_ann_info(idx)
        # print(ann_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """在数据处理管道之后获取测试数据。

        参数:
            idx (int): 数据的索引。

        返回:
            dict: 经过数据处理管道后的测试数据，包含管道引入的新键。
        """

        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)
        self.pre_pipeline(results)
        return self.pipeline(results)

    def __init__(self, split, **kwargs):
        super(GreenlandDataset, self).__init__(
            img_suffix='.png', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
