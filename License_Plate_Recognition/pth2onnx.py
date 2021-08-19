import os
import torch
from model.LPRNet import build_lprnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pthfile = os.path.abspath('weights/Final_LPRNet_model.pth')

lprnet = build_lprnet(lpr_max_len=8, phase=False, class_num=68, dropout_rate=0.5)
lprnet.to(device)
lprnet.load_state_dict(torch.load(pthfile, map_location=torch.device('cpu')))

# 生成张量
dummy_input = torch.randn(1, 3, 24, 94)
dummy_input = dummy_input.to(device)

# 输入输出名
input_names = ['input']
output_names = ['output']

batch_size = 1

# 目的ONNX文件名
onnxfile = 'lprnet.onnx'

torch.onnx.export(lprnet,
                  dummy_input,
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
