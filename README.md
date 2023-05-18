**1. Download MMdetection Framework and Dataset**
####
Please first download mmdetection and MS COCO2017 datasets and make sure that you can run a baseline model successfully.

**2. Train the Teacher Model**
####
Before starting running the distillation codes, you need to train the teacher models via mmdetection. The configs of teachers are as follows:
```
teacher configs:
----fcos_r18_caffe_fpn_gn-head_1x_coco.py
----fcos_r50_caffe_fpn_gn-head_1x_coco.py
----fcos_r101_caffe_fpn_gn-head_1x_coco.py
----retinanet_r18_fpn_1x_coco.py
----retinanet_r50_fpn_1x_coco.py
----retinanet_r101_fpn_1x_coco.py
----faster_rcnn_r18_fpn_1x_coco.py
----faster_rcnn_r50_fpn_1x_coco.py
----faster_rcnn_r101_fpn_1x_coco.py
```
Put them in the corresponding config folders in mmdetection.

**3. Change the Codes of MMdetection**
####

1. Move `ODED_single.py` & `ODED_two.py` in `mmdetection/mmdet/models/detectors/` and change `mmdetection/mmdet/models/detectors/__init__.py`
```
from .ODED_single import ODED_Single
from .ODED_two import ODED_Two

# `__all__` add the follows:
__all__ = [
    'DODED_Single', 'ODED_Two'
]
```
2. Move `adap.py` in `mmdetection/mmdet/models/necks/` and change `mmdetection/mmdet/models/necks/__init__.py`
```
from .adap import ADAP, ADAP_C, ADAP_Residule, ADAP_SINGLE
# `__all__` add the follows:
__all__ = [
    'ADAP', 'ADAP_C', 'ADAP_Residule','ADAP_SINGLE'
]
```
3. Create `ODED` folder in `mmdetection/configs/`. Then, move `ODED_faster_rcnn.py`,`ODED_retinanet.py`, and `ODED_fcos.py` in it.

4. Move `increase_hook.py` in `mmdetection/mmdet/core/utils/`

**4. Train model with ODED**
####
```
python tools/train.py configs/ODED/ODED_faster_rcnn.py --auto-scale-lr  --work-dir work_dirs/
```
**5. Prerequisites**
####
`python 3.7`  `mmdetection`  `PyTorch`
