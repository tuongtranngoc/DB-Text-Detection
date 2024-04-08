from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import torch

import os
import cv2
import argparse

import time
import numpy as np
from pathlib import Path

from src import config as cfg
from src.utils.logger import Logger
from src.utils.data_utils import DataUtils
from src.utils.visualization import Visualization
from src.utils.post_processing import DBPostProcess
from src.models.diff_binarization import DiffBinarization

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

logger = Logger.get_logger("PREDICTION")


class Predictor:
    def __init__(self, args) -> None:
        self.args = args
        self.model = DiffBinarization(pretrained=False)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.model.to(self.args.device)
        self.model.eval()
        self.load_exported_model()
        self.post_process = DBPostProcess(box_thresh=0.5)
        self.image_size = cfg['Train']['dataset']['transforms']['image_shape']
        self.transform = A.Compose([
            A.Resize(self.image_size[1], self.image_size[2]),
            A.Normalize(always_apply=True),
            ToTensorV2()])
        
    def load_exported_model(self):
        if self.args.export_format == 'torchscript':
            logger.info("Loading model for torchscript inference...")
            w = str(self.args.model_path).split('.pth')[0] + '.torchscript'
            self.ts = torch.jit.load(w, map_location=self.args.device)
            self.ts.float()

        if self.args.export_format == 'onnx':
            """https://onnxruntime.ai/docs/api/python/api_summary.html
            """
            logger.info("Loading model for onnx inference...")
            w = str(self.args.model_path).split('.pth')[0] + '.onnx'
            import onnxruntime
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.session = onnxruntime.InferenceSession(w, providers=providers)
            self.binding = self.session.io_binding()
            
        if self.args.export_format == 'tensorrt':
            from torch2trt import TRTModule
            logger.info("Loading model for tensorrt inference ...")
            w = str(self.args.model_path).split('.pth')[0] + '_engine.pth'
            self.model_trt = TRTModule()
            self.model_trt.load_state_dict(torch.load(w, map_location=self.args.device))

        if self.args.export_format == 'paddle':
            logger.info("Loading model for paddle to inference ...")
            w = str(self.args.model_path).split('.pth')[0] + '_paddle_model'
            import paddle.inference as pdi
            w = Path(w)
            if not w.is_file():  # if not *.pdmodel
                w = next(w.rglob('*.pdmodel'))  # get *.pdmodel file from *_paddle_model dir
            config = pdi.Config(str(w), str(w.with_suffix('.pdiparams')))
            config.enable_use_gpu(memory_pool_init_size_mb=2048, device_id=0)
            self.predictor = pdi.create_predictor(config)
            self.input_handle = self.predictor.get_input_handle(self.predictor.get_input_names()[0])
            self.output_names = self.predictor.get_output_names()
        
    def preprocess(self, img_path):
        if os.path.exists(img_path):
            img = cv2.imread(img_path)[..., ::-1]
            img = self.transform(image=img)['image']
            img = img.unsqueeze(0)
            return img
        else:
            Exception("Not exist image path")

    def predict(self, img_path):
        img = self.preprocess(img_path).to(self.args.device)
        _img = img.cpu().detach().numpy()
        save_dir = self.args.save_dir
        # inference with export format
        if self.args.export_format == 'torchscript':
            st = time.time()
            out = self.ts(img)
            logger.info(f"Runtime of {self.args.export_format}: {time.time()-st}")
            
        elif self.args.export_format == 'paddle':
            st = time.time()
            img = img.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(img)
            self.predictor.run()
            out = [self.predictor.get_output_handle(x).copy_to_cpu() for x in self.output_names]
            logger.info(f"Runtime of {self.args.export_format}: {time.time()-st}")
        
        elif self.args.export_format == 'onnx':
            st = time.time()
            img = img.contiguous()
            self.binding.bind_input(
                name=self.session.get_inputs()[0].name,
                device_type=self.args.device,
                device_id=0,
                element_type=np.float32,
                shape=tuple(img.shape),
                buffer_ptr=img.data_ptr(),
            )
            out_shape = [3, 736, 736]
            out = torch.empty(out_shape, dtype=torch.float32, device='cuda:0').contiguous()
            self.binding.bind_output(
                name=self.session.get_outputs()[0].name,
                device_type=self.args.device,
                device_id=0,
                element_type=np.float32,
                shape=tuple(out.shape),
                buffer_ptr=out.data_ptr(),
            )
            self.session.run_with_iobinding(self.binding)
            logger.info(f"Runtime of {self.args.export_format}: {time.time()-st}")

        elif self.args.export_format == 'tensorrt':
            st = time.time()
            out = self.model_trt(img)
            logger.info(f"Runtime of {self.args.export_format}: {time.time()-st}")

        else:
            st = time.time()
            out = self.model(img)
            logger.info(f"Runtime of Pytorch: {time.time()-st}")
        # convert out with export format
        if isinstance(out, list):
            out = out[-1]
        if isinstance(out, np.ndarray):
            out = torch.tensor(out)
        
        out = out.cpu().detach().numpy()
        boxes, scores = self.post_process(_img, out, True)
        boxes, scores = boxes[0], scores[0]
        idxs = np.where(np.array(scores)>args.threshold)[0]
        boxes = [boxes[i].tolist() for i in idxs]
        
        img = DataUtils.image_to_numpy(img)
        img = Visualization.draw_polygon(img, boxes)
        basename = os.path.basename(img_path)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(f"{save_dir}/{basename}", img)


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=cfg['Debug']['prediction'])
    parser.add_argument("--image_path", type=str, default=None, help="Path to image file")
    parser.add_argument("--threshold", type=str, default=0.7, help="Bounding box threshold")
    parser.add_argument("--device", type=str, default='cuda', help="device inference (cuda or cpu)")
    parser.add_argument("--export_format", type=str, default='pytorch', help="Exported format of model to inference")
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'], help="Path to model checkpoint")
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    predictor = Predictor(args)
    for _ in range(10):
        predictor.predict(args.image_path)
    