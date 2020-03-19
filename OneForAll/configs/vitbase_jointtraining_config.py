
from .common import train
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from modeling.meta_arch.multitask_v2 import MultiTaskBatchFuse

# segmentation
from data.transforms.seg_transforms import ResizeStepScaling, RandomPaddingCrop, \
    RandomHorizontalFlip, RandomDistort, Normalize
from data.build_segmentation import build_segmentation_dataset, build_segmentation_trainloader, \
    build_segementation_test_dataset
from evaluation.segmentation_evaluator import SegEvaluator

# classification
from data.build import build_reid_test_loader_lazy
from data.transforms.build import build_transforms_lazy

from data.build_cls import build_hierachical_softmax_train_set, \
    build_hierachical_test_set, build_vehiclemulti_train_loader_lazy
from evaluation.common_cls_evaluator import CommonClasEvaluatorSingleTask

# detection
from data.build_trafficsign import build_cocodet_set, build_cocodet_loader_lazy
from evaluation.cocodet_evaluator import CocoDetEvaluatorSingleTask
from solver.build import build_lr_optimizer_lazy, build_lr_scheduler_lazy

dataloader=OmegaConf.create()
_root = os.getenv("FASTREID_DATASETS", "datasets")

seg_num_classes=19


dataloader.train=L(MultiTaskDataLoader)(
    cfg=dict(sample_mode='batch',),
    task_loaders=L(OrderedDict)(
        segmentation=L(build_segmentation_trainloader)(
            data_set=L(build_segmentation_dataset)(
                    dataset_name="BDD100K",
                    dataset_root=_root + '/track1_train_data/seg/', 
                    transforms=[
                        L(ResizeStepScaling)(min_scale_factor=0.5, max_scale_factor=2.0, scale_step_size=0.25), 
                        L(RandomPaddingCrop)(crop_size=[720, 1280]), 
                        L(RandomHorizontalFlip)(), 
                        L(RandomDistort)(brightness_range=0.4, contrast_range=0.4, saturation_range=0.4),
                        L(Normalize)()],
                    mode='train'),
            total_batch_size=16, 
            worker_num=4, 
            drop_last=True, 
            shuffle=True,
            num_classes=seg_num_classes,
            is_train=True,
        ),

        fgvc=L(build_vehiclemulti_train_loader_lazy)(
            sampler_config={'sampler_name': 'ClassAwareSampler'},
            train_set=L(build_hierachical_softmax_train_set)(
                names = ("FGVCDataset",),
                train_dataset_dir = _root + '/track1_train_data/cls/train/',
                test_dataset_dir = _root + '/track1_train_data/cls/val/',
                train_label = _root + '/track1_train_data/cls/train.txt',
                test_label = _root + '/track1_train_data/cls/val.txt',
                transforms=L(build_transforms_lazy)(
                    is_train=True,
                    size_train=[448, 448],
                    do_rea=True,
                    rea_prob=0.5,
                    do_flip=True,
                    do_autoaug=True,
                    autoaug_prob=0.5,
                    mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                    std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                ),

                num_classes=196,
            ),
            total_batch_size=16,
            num_workers=4,
        ),

        trafficsign=L(build_cocodet_loader_lazy)(
            data_set=L(build_cocodet_set)(
                dataset_name="COCODataSet",
                transforms=[
                    dict(Decode=dict(),),
                    dict(RandomFlip=dict(prob=0.5),),
                    dict(RandomSelect=dict(
                        transforms1=[
                            dict(RandomShortSideResize=dict(
                                short_side_sizes=[480, 512, 544, 576, 608], 
                                max_size=608)
                                ),
                        ],
                        transforms2=[
                            dict(RandomShortSideResize=dict(short_side_sizes=[400, 500, 600]),),
                            dict(RandomSizeCrop=dict(min_size=384, max_size=600),),
                            dict(RandomShortSideResize=dict(
                                short_side_sizes=[480, 512, 544, 576, 608], 
                                max_size=608)
                                ),
                        ],
                    ),),
                    dict(NormalizeImage=dict(
                        is_scale=True, 
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
                        ),
                    dict(NormalizeBox=dict()),
                    dict(BboxXYXY2XYWH=dict()),
                    dict(Permute=dict()),
                ],
                image_dir='train',
                anno_path='train.json',
                dataset_dir= _root + '/track1_train_data/dec/',
                data_fields=['image', 'gt_bbox', 'gt_class', 'is_crowd'],
            ),
            total_batch_size=8,
            num_workers=4,
            batch_transforms=[
                dict(PadMaskBatch=dict(pad_to_stride=-1, return_pad_mask=True),),
            ],
            is_train=True,
            shuffle=True, 
            drop_last=True, 
            collate_batch=False,
        ),
    ),
)

# NOTE
# trian/eval模式用于构建对应的train/eval Dataset, 需提供样本及标签;
# infer模式用于构建InferDataset, 只需提供测试数据, 最终生成结果文件用于提交评测, 在训练时可将该部分代码注释减少不必要评测

dataloader.test = [
    
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            segmentation=L(build_segmentation_trainloader)(
                data_set=L(build_segementation_test_dataset)(
                        dataset_name="BDD100K",
                        dataset_root=_root + '/track1_train_data/seg/', 
                        transforms=[L(Normalize)()],
                        mode='val',
                        is_padding=True),
                total_batch_size=8, 
                worker_num=4, 
                drop_last=False, 
                shuffle=False,
                num_classes=seg_num_classes,
                is_train=False,
            ),
        ),
    ),

    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),