"""
author: luxian
date：  2021/07/16
requirements： see requirements.txt

"""



import argparse
import torch
import os

# 设置命令行参数
parser = argparse.ArgumentParser(description='parameters to convert keras model to tflite')
parser.add_argument('--input_names', default='', help='list or tuple of input names')
parser.add_argument('--output_names', default='', help='list or tuple of output names')
parser.add_argument('--input_shapes', default='', help='a list or tuple for correspond input names, them should be 4D: (1, c, h, w)')
parser.add_argument('--pth_path', default='', help='path to keras model')  
parser.add_argument('--onnx_path', default='', help='file path to save onnx model')  # 必须为文件
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pth_model = torch.load(args.pth_path, map_location=torch.device('cpu'))

# 生成输入张量
dummy_inputs = []
for input_shape in args.input_shapes:
    dummy_input = torch.randn(input_shape)
    dummy_input = dummy_input.to(device)
    dummy_inputs.append(dummy_input)

# 输入输出名
input_names = args.input_names
output_names = args.output_names

batch_size = 1

# 目的ONNX文件名
if os.path.isfile(args.onnx_path):
    onnxfile = args.onnx_path
    torch.onnx.export(pth_model,
                    dummy_inputs,
                    onnxfile,
                    opset_version=9,
                    verbose=True,
                    # export_params=True,
                    do_constant_folding=True,	# 是否执行常量折叠优化
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes={"input": {0: "1"},  # 批处理变量
                                "output": {0: "1"}}
                    )

# import onnxruntime as ort
# import numpy as np
# import cv2
#
# img = cv2.imread("test.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# result = np.zeros(img.shape, dtype=np.float32)
# img = cv2.normalize(img, result, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
# input = np.transpose(img, (2, 0, 1))
#
# onnxfile = 'lprnet.onnx'
#
# ort_session = ort.InferenceSession(onnxfile)
#
# outputs = ort_session.run(
#     None,
#     {'input': input}
# )
# print(outputs)
