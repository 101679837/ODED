_base_ = [
    '../faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
]

model = dict(
    type='Distilling_FRS_Two',

    distill = dict(
        teacher_cfg='./configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
        teacher_model_path1='./model/faster_rcnn_r50_1_epoch_12.pth',
        teacher_model_path2='./model/faster_rcnn_r50_2_epoch_12.pth',
        teacher_model_path3='./model/faster_rcnn_r50_3_epoch_12.pth',
        teacher_model_path4='./model/faster_rcnn_r50_4_epoch_12.pth',
        
        distill_warm_step=500,
        distill_feat_weight=0.002,
        distill_cls_weight=0.1,
        
        stu_feature_adap=dict(
            type='ADAP',
            in_channels=256,
            out_channels=256,
            num=5,
            kernel=3
        ),
    )
)


custom_imports = dict(imports=['mmdet.core.utils.increase_hook'], allow_failed_imports=False)
custom_hooks = [dict(type='NumClassCheckHook'), dict(type='Increase_Hook',)]

seed=520
# find_unused_parameters=True
