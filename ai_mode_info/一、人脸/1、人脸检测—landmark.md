###### 1、[超轻量级通用人脸检测模型](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)

- description

    - desc:  该模型设计是针对边缘计算设备或低算力设备 (如用 ARM 推理) 设计的一款实时超轻量级通用人脸检测模型，旨在能在低算力设备中如用 ARM 进行实时的通用场景的人脸检测推理，同样适用于移动端环境（Android & IOS、PC 环境（CPU & GPU )。
    - keywords: 
        - 超轻量 ( 模型大小仅约为1MB)
        - 算力小 ( 320x240输入下计算量仅为90~109MFlops)
        - 适用于移动端环境
- format
    - model format: .pth文件
    - kind： pretrained
    - dataset：
        - 训练集：使用 Retinaface 提供的清理过的 widerface 标签配合 widerface 数据集生成 VOC 训练集
        - 测试集：WIDER FACE test
- copyright

    - author： Linzaer
    - licence: free of charge
    - published data: 2019
- Comparison of several open source lightweight face detection models:

![image-20210420100309032](https://gitee.com/wonderful4/images/raw/master/imgs/20210420100316.png)