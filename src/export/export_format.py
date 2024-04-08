from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import json
import torch
import argparse
import torch.nn.functional as F

from . import *

logger = Logger.get_logger("EXPORT")


class Exporter:
    def __init__(self, args):
        self.args = args
        self.model = DiffBinarization(pretrained=False)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)['model'])
        self.sample = torch.ones(size=cfg['Train']['dataset']['transforms']['image_shape']).unsqueeze(0).to(self.args.device)
        self.model.to(self.args.device)
        self.model.eval()
    
    def export_torchscript(self):
        """https://pytorch.org/docs/stable/jit.html
        """
        f = str(self.args.model_path).replace('.pth', f'.torchscript')
        logger.info(f'Starting export with torch {torch.__version__}...')
        ts = torch.jit.trace(self.model, self.sample, strict=False)
        logger.info(f'Optimizing for mobile...')
        ts.save(f)
        return f
    
    def export_paddle(self):
        import x2paddle
        from x2paddle.convert import pytorch2paddle
        logger.info(f'Starting export with X2Paddle {x2paddle.__version__}...')
        f = str(self.args.model_path).replace('.pth', f'_paddle_model{os.sep}')
        pytorch2paddle(module=self.model, save_dir=f, jit_type='trace', input_examples=[self.sample])
        return f
        
    def export_onnx(self):
        """https://onnxruntime.ai/docs/api/python/api_summary.html
        """
        import onnx
        logger.info(f'Starting export with onnx {onnx.__version__}...')
        f = str(self.args.model_path).replace('.pth', f'.onnx')
        output_names = ['output0']
        torch.onnx.export(
            self.model,
            self.sample,
            f,
            verbose=False,
            do_constant_folding=True,
            input_names=['images'],
            output_names=output_names,
            dynamic_axes=None
        )
        model_onnx = onnx.load(f)
        onnx.save(model_onnx, f)
        return f

    def export_tensorrt(self):
        """https://github.com/NVIDIA-AI-IOT/torch2trt
        """
        from torch2trt import torch2trt as trt
        f = str(self.args.model_path).replace('.pth', f'_engine.pth')
        logger.info(f"Starting export with tensorrt ...")
        model_trt = trt(self.model, [self.sample], max_workspace_size=32, use_onnx=True, fp16_mode=True)
        torch.save(model_trt.state_dict(), f)
        return f
    
    def __call__(self):
        logger.info("Begining export model ...")
        if self.args.export_format == 'torchscript':
            self.export_torchscript()
        if self.args.export_format == 'paddle':
            self.export_paddle()
        if self.args.export_format == 'onnx':
            self.export_onnx()
        if self.args.export_format == 'tensorrt':
            self.export_tensorrt()
        
    
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=cfg['Train']['checkpoint']['best_path'], help="Path to pytorch model")
    parser.add_argument("--export_format", type=str, default="torchscript", help="Support export formats: torchscript, paddle, TensorRT, ONNX")
    parser.add_argument("--device", type=str, default='cuda', help="Select device for export format")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = cli()
    exporter = Exporter(args)
    exporter()