from .common import train
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from modeling.meta_arch.multitask_v2 import MultiTaskBatchFuse

# segmentation
from data.transforms.seg_transforms import Normalize
from data.build_segmentation import build_segmentation_trainloader, \
    build_segementation_test_dataset
from evaluation.segmentation_evaluator import SegEvaluatorInfer

# classification
from data.build import build_reid_test_loader_lazy
from data.transforms.build import build_transforms_lazy

from data.build_cls import build_hierachical_test_set
from evaluation.common_cls_evaluator import CommonClasEvaluatorSingleTaskInfer

# detection
from data.build_trafficsign import build_cocodet_set, build_cocodet_loader_lazy
from evaluation.cocodet_evaluator import CocoDetEvaluatorSingleTaskInfer

dataloader=OmegaConf.create()
_root = os.getenv("FASTREID_DATASETS", "datasets")

seg_num_classes=19

# NOTE
# trian/eval模式用于构建对应的train/eval Dataset, 需提供样本及标签;
# infer模式用于构建InferDataset, 只需提供测试数据, 最终生成结果文件用于提交评测, 在训练时可将该部分代码注释减少不必要评测

dataloader.test = [
    
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            segmentation=L(build_segmentation_trainloader)(
                data_set=L(build_segementation_test_dataset)(
                        dataset_name="InferDataset",
                        dataset_root=_root + '/track1_test_data/seg/', 
                        transforms=[L(Normalize)()],
                        mode='test',
                        is_padding=True),
                total_batch_size=8, 
                worker_num=4, 
                drop_last=False, 
                shuffle=False,
                num_classes=seg_num_classes,
               