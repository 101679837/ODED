from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .two_stage import TwoStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F

import numpy as np

@DETECTORS.register_module()
class Distilling_FRS_Two(TwoStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_FRS_Two, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        from mmdet.apis.inference import init_detector

        self.device = torch.cuda.current_device()
        self.teacher1 = init_detector(distill.teacher_cfg, \
                        distill.teacher_model_path1, self.device)
                        
        self.teacher2 = init_detector(distill.teacher_cfg, \
                        distill.teacher_model_path2, self.device)
                        
        self.teacher3 = init_detector(distill.teacher_cfg, \
                        distill.teacher_model_path3, self.device)
                        
        self.teacher4 = init_detector(distill.teacher_cfg, \
                        distill.teacher_model_path4, self.device)
                        
        self.stu_feature_adap = build_neck(distill.stu_feature_adap)

        self.distill_feat_weight = distill.get("distill_feat_weight",0)
        self.distill_cls_weight = distill.get("distill_cls_weight",0)

        for m in self.teacher1.modules():
            for param in m.parameters():
                param.requires_grad = False
        self.distill_warm_step = distill.distill_warm_step

        self.debug = distill.get("debug",False)
            
    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        x = self.extract_feat(img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        
        stu_feature_adap = self.stu_feature_adap(x)
        y1 = self.teacher1.extract_feat(img)
        y2 = self.teacher2.extract_feat(img)
        y3 = self.teacher3.extract_feat(img)
        y4 = self.teacher4.extract_feat(img)
        
        
        y11 = list(y1)
        y22 = list(y2)
        y33 = list(y3)
        y44 = list(y4)

        
        
        y00 = list(y1) 
         

            #feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
        #y =  (y1+y2+y3+y4)/4

        stu_bbox_outs = self.rpn_head(x)
        stu_cls_score = stu_bbox_outs[0]
        layers = len(stu_cls_score)
        
        
        for layer in range(layers):
            y00[layer] =  (y11[layer]+y22[layer]+y33[layer]+y44[layer])/4
            #print('LLLLL::::', y11[layer])
            
        y = tuple(y00) 
        #print('LLLLL::::', layer)
        
        stu_bbox_outs = self.rpn_head(y)
        stu_cls_score = stu_bbox_outs[0]
        
        

        tea_bbox_outs1 = self.teacher1.rpn_head(y)
        tea_bbox_outs2 = self.teacher2.rpn_head(y)
        tea_bbox_outs3 = self.teacher3.rpn_head(y)
        tea_bbox_outs4 = self.teacher4.rpn_head(y)
        
        ll =  len(tea_bbox_outs1[0])
        
        
        tea_bbox_outs11 = list(tea_bbox_outs1[0])
        tea_bbox_outs22 = list(tea_bbox_outs2[0])
        tea_bbox_outs33 = list(tea_bbox_outs3[0])
        tea_bbox_outs44 = list(tea_bbox_outs4[0])
        
        tea_bbox_outs00 = list(tea_bbox_outs1[0])
        
        for l in range(ll):
            tea_bbox_outs00[l] =  (tea_bbox_outs11[l]+tea_bbox_outs22[l]+tea_bbox_outs33[l]+tea_bbox_outs44[l])/4
            #print('LLLLL::::', y11[layer])
        
        
        #print('LLLLL::::', (tea_bbox_outs11))
        #print('LLLLL::::', (tea_bbox_outs22))
        #print('LLLLL::::', len(tea_bbox_outs33))
        #print('LLLLL::::', len(tea_bbox_outs44))
        #print('LLLLL::::', len(tea_bbox_outs33))
        '''

        tea_bbox_outs11= torch.tensor([item.cpu().detach().numpy() for item in tea_bbox_outs11]).cuda()
        tea_bbox_outs22= torch.tensor([item.cpu().detach().numpy() for item in tea_bbox_outs22]).cuda()
        tea_bbox_outs33= torch.tensor([item.cpu().detach().numpy() for item in tea_bbox_outs33]).cuda()
        tea_bbox_outs44= torch.tensor([item.cpu().detach().numpy() for item in tea_bbox_outs44]).cuda()
        
        tea_bbox_outs000=  (np.array(tea_bbox_outs11) + np.array(tea_bbox_outs22) + np.array(tea_bbox_outs33) + np.array(tea_bbox_outs44))/4
        
        
        #tea_bbox_outs00 = array.tolist(tea_bbox_outs000)
        
        #tea_bbox_outs = tuple(tea_bbox_outs00)
        '''
        tea_cls_score = tuple(tea_bbox_outs00)

        
        distill_feat_loss, distill_cls_loss = 0, 0

        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            mask = mask.detach()

            #feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            #cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')
            
            feat_loss = torch.abs(y[layer] - stu_feature_adap[layer])
            cls_loss = torch.pow(stu_cls_score_sigmoid-tea_cls_score_sigmoid,2).sigmoid()

            distill_feat_loss += (feat_loss * mask[:,None,:,:]).sum() / mask.sum()/10
            distill_cls_loss +=  (cls_loss * mask[:,None,:,:]).sum() / mask.sum()/10
            # breakpoint()

        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        distill_cls_loss = distill_cls_loss * self.distill_cls_weight

        if self.debug:
            print(self._inner_iter, distill_feat_loss, distill_cls_loss)

        if self.distill_warm_step > self.iter:
            distill_feat_loss = (self.iter / self.distill_warm_step) * distill_feat_loss
            distill_cls_loss = (self.iter / self.distill_warm_step) * distill_cls_loss

        if self.distill_feat_weight:
            losses.update({"distill_feat_loss":distill_feat_loss})
        if self.distill_cls_weight:
            losses.update({"distill_cls_loss":distill_cls_loss})

        return losses
