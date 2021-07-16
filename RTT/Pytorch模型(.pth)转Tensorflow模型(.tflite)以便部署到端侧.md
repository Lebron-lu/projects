# **Pytorch模型(.pth)转Tensorflow模型(.tflite)以便部署到端侧**设备

### 总结

Pytorch框架易于模型训练，而Tensorflow的.tflite模型文件更适合与端侧部署。所以两者存在模型文件类型转换的需要。.pth模型转.tflite模型的全部流程：

![image.png](https://gitee.com/wonderful4/images/raw/master/imgs/20210611165432.png)

### 环境配置

配置环境是前提！此次转换流程所需要的包安装如下：

> torch==1.5.1
>
> torchvision==0.6.1
>
> tf-models-nightly==2.3.0.dev20200811
>
> onnx==1.7.0
>
> onnxruntime==1.7.0
>
> onnx-tf==1.7.0
>
> tensorflow-addons==0.11.2

简单的环境配置方法，将上述包写成requirements.txt文件：

![image-20210611165705492](https://gitee.com/wonderful4/images/raw/master/imgs/20210611165705.png)

cd到.txt文件所在目录，然后输入以下命令行即可安装全部所需包(亲测可安装)：

```bash
pip install -r requirements.txt
```

![image-20210611170134599](https://gitee.com/wonderful4/images/raw/master/imgs/20210611170134.png)

### 1、.pth转.onnx

```python
import io
import torch 
import torch.onnx
import model
pth_path = './LPR_net_Acc_85.pth'  #需要转换的.pth模型,注意这里的.pth文件应包含参数和网络图结构
model = model()
model = torch.load(pth_path, map_location='cpu')  #加载.pth模型
dummy_input = torch.randn(1, 3, 24, 94) #该数据的形状为模型输入的shape
input_names = ['input']  #模型输入节点名称
output_names = ['output']  #模型输出节点名称。可以用print(model)查看节点名称
output_model_path = './lprnet_acc85.onnx' #输出.onnx模型路径
torch.onnx.export(model, 
                  dummy_input, 
                  output_model_path, 
                  verbose=True, 
                  input_names=input_names, 
                  output_names=output_names)
```

输出.onnx文件

![image-20210611170312803](https://gitee.com/wonderful4/images/raw/master/imgs/20210611170312.png)

在这里我们需要注意的是，需要转换的.pth模型文件，应该包含参数和网络图结构，如果只包含参数是不能用来进行转换的。所以，在保存.pth文件时使用  **torch.save(model)**  而不是  **torch.save(model.state_dict)**，前者是保存模型参数和网络图结构，后者仅仅保存模型参数。

### 2、.onnx转.pb

```python
from onnx_tf.backend import prepare
import onnx
pb_path = './pbmodel' #输出.pb等三个文件的储存路径
onnx_path = './lprnet_acc85.onnx' #.onnx模型文件的路径
onnx_model = onnx.load(onnx_path) #加载onnx模型
tf_rep = prepare(onnx_model) #创建tensorflowrep对象
tf_rep.export_graph(pb_path) #输出.pb文件
```

在pb_path文件夹中输出.pb等三个文件：

![image-20210611170423426](https://gitee.com/wonderful4/images/raw/master/imgs/20210611170423.png)

### 3、.pb转.tflite

```python
import tensorflow as tf
pb_path = "./pbmodel" 
tflite_path = "lprnet.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(pb_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tf_lite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)
```

输出.tflite模型文件：

![image-20210611170505178](https://gitee.com/wonderful4/images/raw/master/imgs/20210611170505.png)

### 部署

至此，我们已经将pytorch模型文件转为tensorflow中的tflite模型文件，这样方便我们将其部署到端侧设备。转换完成后，我们可以使用RT-AK工具一键将其部署。