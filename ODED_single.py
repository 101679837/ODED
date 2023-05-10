from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .single_stage import SingleStageDetector
from mmdet.core.bbox.iou_calculators import *
import torch
import torch.nn.functional as F
# from mmdet.apis.inference import init_detector

@DETECTORS.register_module()
class Distilling_FRS_Single(SingleStageDetector):

    def __init__(self,
                 backbone=None,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 distill=None,):
        super(Distilling_FRS_Single, self).__init__(backbone, neck, bbox_head, train_cfg,
                                  test_cfg, pretrained)
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
                      gt_bboxes_ignore=None):

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        
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
        y = tuple(y00)  

            #feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
        #y =  (y1+y2+y3+y4)/4

        stu_bbox_outs = self.bbox_head(x)
        stu_cls_score = stu_bbox_outs[0]
        layers = len(stu_cls_score)
        
        
        for layer in range(layers):
            y00[layer] =  (y11[layer]+y22[layer]+y33[layer]+y44[layer])/4
            #print('LLLLL::::', y11[layer])
        
        y = tuple(y00)  
   
        

        #print('LLLLL::::', layer)
        

        tea_bbox_outs1 = self.teacher1.bbox_head(y)
        tea_bbox_outs2 = self.teacher2.bbox_head(y)
        tea_bbox_outs3 = self.teacher3.bbox_head(y)
        tea_bbox_outs4 = self.teacher4.bbox_head(y)
        
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
        
        tea_cls_score1 = tuple(tea_bbox_outs11)
        tea_cls_score2 = tuple(tea_bbox_outs22)
        tea_cls_score3 = tuple(tea_bbox_outs33)
        tea_cls_score4 = tuple(tea_bbox_outs44)
        '''
        stu_bbox_outs = self.bbox_head(y)
        stu_cls_score = stu_bbox_outs[0]
        
        tea_cls_score = tuple(tea_bbox_outs00)
        
        distill_feat_loss, distill_cls_loss = 0, 0
        
        #print("LLLLLLLLLL", layers)

        for layer in range(layers):
            stu_cls_score_sigmoid = stu_cls_score[layer].sigmoid()
            tea_cls_score_sigmoid = tea_cls_score[layer].sigmoid()
            mask = torch.max(tea_cls_score_sigmoid, dim=1).values
            mask = mask.detach()

            #feat_loss = F.l1_loss(y[layer].sigmoid(), stu_feature_adap[layer].sigmoid())
            #print('AAAAAAAAAAA::::feat_loss', feat_loss)
            
            #cls_loss = F.mse_loss(stu_cls_score_sigmoid, tea_cls_score_sigmoid).sigmoid()
            
            #print('AAAAAAAAAAA::::cls_loss', cls_loss)
            
            
            #feat_loss = torch.pow((y[layer] - stu_feature_adap[layer]), 2)
            #cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none').sigmoid()
            feat_loss = torch.abs(y[layer].sigmoid() - stu_feature_adap[layer].sigmoid()).sigmoid()
            #print('BBBBBBBBBBBBBBBBBBB::::feat_loss', feat_loss)
            #cls_loss = F.binary_cross_entropy(stu_cls_score_sigmoid, tea_cls_score_sigmoid,reduction='none')
            
            #feat_loss = torch.abs(y1[layer] - stu_feature_adap[layer])
            #print('BBBBBBBBBBBBBBBBBBB::::feat_loss', feat_loss)
            cls_loss = torch.pow(stu_cls_score[layer] - tea_cls_score[layer],2).sigmoid()/40
            #print('BBBBBBBBBBBBBBBBB::::cls_loss', cls_loss)

            distill_feat_loss += (feat_loss * mask[:,None,:,:]).sum() / mask.sum() /10
            
            distill_cls_loss +=  (cls_loss * mask[:,None,:,:]).sum() / mask.sum() /10
            
        
        #print("AAAAAAAAAAAAAA",distill_feat_loss)
        #print("BBBBBBBBBBBBBB",distill_feat_loss)
        distill_feat_loss = distill_feat_loss * self.distill_feat_weight
        distill_cls_loss = distill_cls_loss * self.distill_cls_weight

        if self.debug:
            # if self._inner_iter == 10:
            #     breakpoint()
            print(self._inner_iter, distill_feat_loss, distill_cls_loss)

        if self.distill_warm_step > self.iter:
            distill_feat_loss = (self.iter / self.distill_warm_step) * distill_feat_loss
            distill_cls_loss = (self.iter / self.distill_warm_step) * distill_cls_loss

        if self.distill_feat_weight:
            losses.update({"distill_feat_loss":distill_feat_loss})
        if self.distill_cls_weight:
            losses.update({"distill_cls_loss":distill_cls_loss})

        return losses
