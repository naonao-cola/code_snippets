#!/usr/bin/env python3
# -*- coding:utf-8 -*-

'''
python3.7 trt_yolox.py -c ./models/base_model.pth -ih 256 -iw 256 --model_half --input_half -mbs 1 -bs 1

Note: When C++ TensorRT inference, “input is float32, model is half (fp16)” is permitted.
'''

import os
import os.path as osp
import argparse

import tensorrt as trt
import torch
from torch2trt import torch2trt
from tvdl.detection import YOLOX


def make_parser():
    parser = argparse.ArgumentParser("YOLOX TensorRT deploy")
    parser.add_argument("-c", "--ckpt_path", type=str, help="path and model ckpt  file")
    parser.add_argument("-ih", "--input_h", type=int, help="input height")
    parser.add_argument("-iw", "--input_w", type=int, help="input width")
    parser.add_argument("-ihf", '--input_half', action="store_true", help='input images use half(fp16)')
    parser.add_argument("-mhf", '--model_half', action="store_true", help='model paramaters use half(fp16)')
    parser.add_argument("-bs", '--batchsize', type=int, default=1, help='batch size in detect')
    parser.add_argument("-mbs", '--max_batchsize', type=int, default=1, help='max batch size in detect')
    parser.add_argument("-o", '--out_dir', type=str, default=None, help='output model path. if None, osp.dirname(ckpt_path')
    parser.add_argument("-w", '--workspace', type=int, default=32, help='max workspace size in detect')
    return parser


def main():
    args = make_parser().parse_args()
    if args.ckpt_path is None:
        ckpt_file = "./base_model.pth"
    else:
        ckpt_file = args.ckpt_path
    input_shape = (args.batchsize, 3, args.input_h, args.input_w)

    Suffix = "i_FP16" if args.input_half else "i_FP32"
    Suffix += "-m_FP16" if args.model_half else "m_FP32"
    out_dir = osp.dirname(ckpt_file) if args.out_dir is None else args.out_dir
    out_pth = os.path.join(out_dir, "model_trt-{}.pth".format(Suffix))
    engine_file = os.path.join(out_dir, "model_trt-{}.engine".format(Suffix))

    print(">>> input_shape: (N,C,H,W) = ", input_shape)
    print(">>> max_workspace_size: ", (1 << args.workspace))
    print(">>> will save pth: {}".format(out_pth))
    print(">>> will save engine_file: {}".format(engine_file))

    print("loading from ", ckpt_file)
    model = YOLOX.load_from_checkpoint(ckpt_file)
    print("\nload checkpoint done.")
    model.eval()
    model.cuda()
    if args.input_half:
        model = model.half()
    # special for TensorRT
    model.head.decode_in_inference = False

    x = torch.ones(*input_shape).cuda()
    if args.input_half:
        x = x.half()

    print("\nConverting ...")
    model_trt = torch2trt(
        model,
        [x],
        fp16_mode=args.model_half,
        log_level=trt.Logger.INFO, # trt.Logger.VERBOSE,
        max_workspace_size=(1 << args.workspace),
        max_batch_size=args.max_batchsize,
    )

    torch.save(model_trt.state_dict(), out_pth)
    print("\nConverted TensorRT model done. file: {}".format(out_pth))
    with open(engine_file, "wb") as f:
        f.write(model_trt.engine.serialize())
    print("\nConverted TensorRT model engine file is saved for C++ inference. out engine file: {}".format(engine_file))


if __name__ == "__main__":
    main()
