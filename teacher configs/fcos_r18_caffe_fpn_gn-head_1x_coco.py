_base_ = './fcos_r50_caffe_fpn_gn-head_1x_coco.py'  # noqa

model = dict(
    backbone=dict(
        depth=18,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')),
    neck=dict(in_channels=[64, 128, 256, 512]))