# coding=utf-8
'''
@ Summary: 
@ Update:  

@ file:    keras2tflite_dynamic.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/7/1 下午7:10
'''
import os
import cv2
import time
import numpy as np
import tensorflow as tf
from pathlib import Path

test_path = "../Images/test.jpg"
img_shape = (64, 64)
keras_file = '../Models/20210701.h5'
tflite_file = Path("../Models/Tflites/fire_dynamic.tflite")

def main():
    # 单个测试样本数据
    image = cv2.imread(test_path)
    image = cv2.resize(image, img_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_bn = image.astype("float32") / 255.0
    image_bn = np.expand_dims(image_bn, axis=0)


    # 恢复 keras 模型，并预测
    model = tf.keras.models.load_model(keras_file)

    # 动态量化 dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    tflite_file.write_bytes(tflite_model)

    # tflite 模型推理
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    interpreter.set_tensor(input_details['index'], image_bn)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details['index'])
    print(f"prediction: {output_data}")
    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    print("model size: {:.2f} MB".format(os.path.getsize(tflite_file)/1024/1024))


if __name__ == "__main__":
    main()