"""
author: luxian
date：  2021/07/16
requirement： tensorflow2

"""



import tensorflow as tf
import argparse
import os

# 设置命令行参数
parser = argparse.ArgumentParser(description='parameters to convert keras model to tflite')
parser.add_argument('--keras_path', default='', help='path to keras model')  
parser.add_argument('--tflite_path', default='', help='path to save tflite model: file or content')  # 可以为文件或者根目录
parser.add_argument('--tflite_name', default='network', help='the name of tflite model after converting') 
args = parser.parse_args()

# converter for keras to tflite
cvtmodel = tf.keras.models.load_model(args.keras_path)
converter = tf.lite.TFLiteConverter.from_keras_model(cvtmodel)
tflite_model = converter.convert()

# 写进tflite文件
if os.path.isfile(args.tflite_path):
    with open(argparse.tflite_path, "wb") as f:
        f.write(tflite_model)
elif os.path.isdir(args.tflite_path):
    with open(os.path.join(argparse.tflite_path, args.tflite_name + '.tflite'), "wb") as f:
        f.write(tflite_model)
print("\n ********** Successful to convert tflite model! ********** \n")