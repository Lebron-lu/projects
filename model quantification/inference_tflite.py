# coding=utf-8
'''
@ Summary: 
@ Update:  

@ file:    inference_tflite.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/6/30 下午5:07
'''
import cv2
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image):
    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # check the type of the input tensor
    floating_model = input_details['dtype'] == np.float32

    # NxHxWxC, H:1, W:2
    height, width = input_details['shape'][1:3]
    img = cv2.imread(test_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))

    # add N dim
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = np.float32(input_data) / 255.0


    interpreter.set_tensor(input_details['index'], input_data)

    start_time = time.time()
    interpreter.invoke()
    stop_time = time.time()

    output_data = interpreter.get_tensor(output_details['index'])
    print(f"prediction: {output_data}")
    print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

    # if input_details['dtype'] == np.uint8:
    #     input_scale, input_zero_point = input_details["quantization"]
    #     test_image = test_image / input_scale + input_zero_point
    # test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    # interpreter.set_tensor(input_details["index"], test_image)
    # interpreter.invoke()
    # output = interpreter.get_tensor(output_details["index"])
    # print(output)
    #
    # return output


def main():
    # tflite_file = "./mnist_tflite_models/mnist_model_quant.tflite"
    tflite_file = "../Models/Tflites/fire_raw.tflite"
    test_image = "test2.jpg"

    # test_image = read_image(test_image)
    run_tflite_model(tflite_file, test_image)


if __name__ == "__main__":
    main()