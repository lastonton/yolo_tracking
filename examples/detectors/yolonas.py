# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from boxmot.utils import logger as LOGGER
from examples.detectors.yolo_interface import YoloInterface


class YoloNASStrategy(YoloInterface):
    pt = False
    stride = 32
    fp16 = False
    triton = False
    names = {
        0: 'người', 1: 'xe đạp', 2: 'ô tô', 3: 'xe máy', 4: 'máy bay', 5: 'xe buýt',
        6: 'tàu hỏa', 7: 'xe tải', 8: 'thuyền', 9: 'đèn tín hiệu giao thông', 10: 'vòi chữa cháy',
        11: 'biển báo dừng xe', 12: 'đồng hồ tính tiền đỗ xe', 13: 'ghế dài', 14: 'con chim', 15: 'con mèo',
        16: 'con chó', 17: 'con ngựa', 18: 'con cừu', 19: 'con bò', 20: 'con voi',
        21: 'con gấu', 22: 'con ngựa vằn', 23: 'con hươu cao cổ', 24: 'balo', 25: 'cây dù',
        26: 'túi xách', 27: 'cà vạt', 28: 'vali', 29: 'đĩa frisbee', 30: 'ván trượt tuyết',
        31: 'ván trượt tuyết ván đôi', 32: 'quả bóng thể thao', 33: 'con diều', 34: 'gậy bóng chày', 35: 'găng tay bóng chày',
        36: 'ván trượt', 37: 'ván lướt sóng', 38: 'vợt tennis', 39: 'chai', 40: 'ly rượu vang',
        41: 'tách', 42: 'nĩa', 43: 'dao', 44: 'muỗng', 45: 'tô',
        46: 'chuối', 47: 'táo', 48: 'sandwich', 49: 'cam', 50: 'bông cải xanh',
        51: 'cà rốt', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'bánh ngọt',
        56: 'ghế', 57: 'ghế sofa', 58: 'cây cảnh trong chậu', 59: 'giường', 60: 'bàn ăn',
        61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'chuột máy tính', 65: 'điều khiển',
        66: 'bàn phím', 67: 'điện thoại di động', 68: 'lò vi sóng', 69: 'lò nướng', 70: 'máy nướng bánh mì',
        71: 'bồn rửa bát', 72: 'tủ lạnh', 73: 'sách', 74: 'đồng hồ', 75: 'lọ hoa',
        76: 'kéo', 77: 'gấu bông', 78: 'máy sấy tóc', 79: 'bàn chải đánh răng'
    }

    def __init__(self, model, device, args):
        self.args = args

        avail_models = [x.lower() for x in list(Models.__dict__.keys())]
        model_type = self.get_model_from_weigths(avail_models, model)

        LOGGER.info(f'Loading {model_type} with {str(model)}')
        if not model.exists() and model.stem == model_type:
            LOGGER.info('Downloading pretrained weights...')
            self.model = models.get(
                model_type,
                pretrained_weights="coco"
            ).to(device)
        else:
            self.model = models.get(
                model_type,
                num_classes=-1,  # set your num classes
                checkpoint_path=str(model)
            ).to(device)

        self.device = device

    @torch.no_grad()
    def __call__(self, im, augment, visualize):

        im = im[0].permute(1, 2, 0).cpu().numpy() * 255

        with torch.no_grad():
            preds = self.model.predict(
                im,
                iou=0.5,
                conf=0.7,
                fuse_model=False
            )[0].prediction

        preds = np.concatenate(
            [
                preds.bboxes_xyxy,
                preds.confidence[:, np.newaxis],
                preds.labels[:, np.newaxis]
            ], axis=1
        )

        preds = torch.from_numpy(preds).unsqueeze(0)

        return preds

    def warmup(self, imgsz):
        pass

    def postprocess(self, path, preds, im, im0s):

        results = []
        for i, pred in enumerate(preds):

            if pred is None:
                pred = torch.empty((0, 6))
                r = Results(
                    path=path,
                    boxes=pred,
                    orig_img=im0s[i],
                    names=self.names
                )
                results.append(r)
            else:

                pred[:, :4] = ops.scale_boxes(im.shape[2:], pred[:, :4], im0s[i].shape)

                # filter boxes by classes
                if self.args.classes:
                    pred = pred[torch.isin(pred[:, 5].cpu(), torch.as_tensor(self.args.classes))]

                r = Results(
                    path=path,
                    boxes=pred,
                    orig_img=im0s[i],
                    names=self.names
                )
            results.append(r)
        return results
