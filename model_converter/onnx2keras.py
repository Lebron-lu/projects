"""
author: luxian
date：  2021/07/16
requirements： see requirements.txt

"""



from onnx_tf.backend import prepare
import onnx
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='parameters to convert onnx model to keras')
parser.add_argument('--onnx_path', default='', help='path to onnx model')  
parser.add_argument('--keras_path', default='', help='content to save keras model')  # 一定要为目录
args = parser.parse_args()

# onnx to pb
onnx_model = onnx.load(args.onnx_path)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(args.keras_path)
print("\n ********** Successful to convert keras model! ********** \n")