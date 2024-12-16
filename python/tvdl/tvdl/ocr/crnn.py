'''
Copyright (C) 2023 TuringVision

Implementation of Crnn
'''
import cv2
import math
import time
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
from ..common import OcrRecModelBase


__all__ = ["Crnn"]


char_nums = {
    "en": 37,
    "ch": 6625,
}


class Crnn(OcrRecModelBase):
    def __init__(self,
        img_c: int = 3,
        magnitude: str = "server",
        character_type: str = 'en',
        use_space_char: bool = False,
        max_text_length: int = 25,
        batch_number: int = 4,
        rec_image_shape=[3, 32, 320],
        limited_max_width: int = 1280,
        limited_min_width: int = 16,
        ):
        """
        the module for OCR. Support english and chinese now.
        args:
            img_c: image channle, default 3.
            magnitude: The size of model, value "server" or "mobile".
            character_type: Character type, support 'en'、'ch' now.
            use_space_char: User or not use space char, value True, False.
            max_text_length: The max recognition char lenght.
        """
        super().__init__()
        self.hparams.update({'img_c': img_c,
                             'magnitude': magnitude,
                             'character_type': character_type,
                             'use_space_char': use_space_char,
                             'max_text_length': max_text_length,
                             'batch_number': batch_number,
                             'rec_image_shape': rec_image_shape,
                             'limited_max_width': limited_max_width,
                             'limited_min_width': limited_min_width,
                             })
        self.build_model()
        if magnitude == "mobile":
            self.load_state_dict(torch.load(osp.join(osp.dirname(__file__), "models/ch_mobile_mv3_f16_rec.pth")))
        else:
            self.load_state_dict(torch.load(osp.join(osp.dirname(__file__), "models/ch_server_r34_f16_rec.pth")))

    def build_model(self):
        from .backbone.rec_resnet_vd import ResNet
        from .neck.rnn import SequenceEncoder
        from .head.rec_ctc_head import CTCHead
        from .postprocess.rec_postprocess import CTCLabelDecode
        from .backbone.rec_mobilenet_v3 import MobileNetV3

        p = self.hparams
        # build backbone
        hidden_size = 256
        fc_decay = 0.00004
        img_c = 3 if p.img_c == 1 else p.img_c
        if p.magnitude == "server":
            self.backbone = ResNet(in_channels=img_c, layers=34)
        elif p.magnitude == "mobile":
            self.backbone = MobileNetV3(in_channels=img_c, model_name="small", small_stride=[1, 2, 2, 2])
            hidden_size = 48
            fc_decay = 0.00001
        else:
            raise(Exception("Error magnitude value"))

        in_channels = self.backbone.out_channels

        # build neck
        self.neck = SequenceEncoder(in_channels, "rnn", hidden_size=hidden_size)
        in_channels = self.neck.out_channels

        # build head
        self.head = CTCHead(in_channels, out_channels=char_nums[p.character_type], fc_decay=fc_decay)
        self._initialize_weights()

        # build post op.
        self.postprocess_op = CTCLabelDecode(character_type=p.character_type, use_space_char=p.use_space_char)

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x

    def predict(self, img_list):
        p = self.hparams
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))

        # rec_res = []
        rec_res = [['', 0.0]] * img_num
        batch_num = p.batch_number
        elapse = 0
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            max_wh_ratio = 0

            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)

            for ino in range(beg_img_no, end_img_no):
                norm_img = self.resize_norm_img(img_list[indices[ino]],
                                                max_wh_ratio)
                norm_img = norm_img[np.newaxis, :]
                norm_img_batch.append(norm_img)

            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()

            starttime = time.time()
            with torch.no_grad():
                inp = torch.from_numpy(norm_img_batch)
                inp = inp.to(self.device).to(self.dtype)
                prob_out = self.forward(inp)
            preds = prob_out.cpu().numpy()

            rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]
            elapse += time.time() - starttime
        return rec_res, elapse

    def resize_norm_img(self, img, max_wh_ratio):
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        p = self.hparams
        imgC, imgH, imgW = p.rec_image_shape
        assert imgC == img.shape[2]
        max_wh_ratio = max(max_wh_ratio, imgW / imgH)
        imgW = int((32 * max_wh_ratio))
        imgW = max(min(imgW, p.limited_max_width), p.limited_min_width)
        h, w = img.shape[:2]
        ratio = w / float(h)
        ratio_imgH = math.ceil(imgH * ratio)
        ratio_imgH = max(ratio_imgH, p.limited_min_width)
        if ratio_imgH > imgW:
            resized_w = imgW
        else:
            resized_w = int(ratio_imgH)
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im