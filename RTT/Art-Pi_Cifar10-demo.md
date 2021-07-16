# 使用 RT-AK 部署 Cifar10 模型至 Art-Pi

[TOC]

> github：https://github.com/EdgeAIWithRTT/Project7-Cifar10_Cube_Art-Pi

## RT-AK 及 Art-Pi 简介

> 目前该项目为 RT-AK 的示例 Demo，基于 ART-PI 硬件平台和 Cifar10 数据集。

- `RT-AK`: `RT-Thread AI Toolkit`，RT-Thread AI 套件。

`RT-AK` 是 `RT-Thread` 团队为 `RT-Thread` 实时操作系统所开发的 `AI` 套件，能够一键将 `AI` 模型部署到 RT-Thread 项目中，让用户可以 在统一的 API 之上进行业务代码开发，又能在目标平台上获极致优化的性能，从而更简单方便地开发端侧 AI 应用程序。

在 RT-AK 支持下，仅需要一行命令，即可将 AI 模型部署到 RT-Thread 系统中：

```plaintext
$ python rt_ai_tools.py --model xxx...
```

------

**我们将致力于降低嵌入式 AI 落地的难度和门槛**。

- `Art-Pi`：

    ![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709143531.png)

## 使用 Tensorflow2.5.0 进行模型量化

​		我们已提供 keras 模型量化为 tflite 模型的代码 [链接](https://git.rt-thread.com/luxian/art-pi_cifar10_without_lcd/-/blob/v0.1.0/model/keras2tflite_int.py)。注意一定要下载 tensorflow2.5.0 及以上版本，否则会转化失败。该链接中还有量化后的 tflite 的推理代码。目录如下：

![image-20210709143810697](https://gitee.com/wonderful4/images/raw/master/imgs/20210709143810.png)

## 一. 将 Cifar10 分类模型 部署至 Art-Pi (不搭建LCD) 

### 1. 项目总结

​		该项目不需要任何驱动，仅将 AI 模型部署至 Art-Pi 硬件平台，并利用其元件进行 AI 模型推理，最后在终端输出模型推理结果相关信息。

### 2. 空项目工程创建

​		该项目是基于 ART-PI 的模板工程，初始工程来于 RT-Thread 新建工程

​		经过 RT-AK 转换之后得到的一个完整的项目工程

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709143942.png)

### 3. RT-AK 的使用

​		cd 到 aitools.py 所在目录，打开 cmd 命令行窗口，输入以下命令即可获得集成 AI 模型的完整项目工程。

```bash
python aitools.py --project=C:\Users\Admin\Desktop\Art-Pi_cifar10_without_lcd 
--model=C:\Users\Admin\Desktop\Art-pi\Art-Pi_cifar10\model\cifar10_int8.tflite 
--model_name=cifar10 
--platform=stm32 
--ext_tools=C:\Users\Admin\Desktop\RTAK-tools\stm32ai-windows-5.2.0\windows # x_cube_ai 工具
```

### 4. 集成 AI 模型的项目工程的编译和烧录

​		这里我们使用 RT-Thread Studio 进行编译和烧录

![img](https://cdn.nlark.com/yuque/0/2021/png/21594242/1625800399211-bf2389fb-4513-47ad-8a83-f35f17502497.png?x-oss-process=image%2Fresize%2Cw_748)

### 5. AI 应用开发

我们提供了一个实例代码 [cifar10_app.c](https://github.com/EdgeAIWithRTT/Project7-Cifar10_Cube_Art-Pi/blob/v0.1.0/Art-Pi_cifar10_without_lcd/applications/cifar10_app.c)，在终端命令行中输入 cifat10_app.c，即可获得输出

![img](https://cdn.nlark.com/yuque/0/2021/png/21594242/1625800498800-107e0e3f-703b-4ab4-bbaa-1d5c6304122d.png)

### 6. 自定义数据

​		该项目提供了自定义数据生成代码 [save_img.py](https://github.com/EdgeAIWithRTT/Project7-Cifar10_Cube_Art-Pi/blob/master/cifar10_data/save_img.py)。

## 二. 将 Cifar10 分类模型 部署至 Art-Pi, 并搭建LCD

### 1. 项目总结

​		该项目应用了 LCD 驱动相关代码，使用 RT-AK 将 AI 模型部署至 Art-Pi 硬件平台之后，利用其元件进行 AI 模型推理， 并对模型输出结果进行处理，最后将模型输入图片和输出相关信息显示到 LCD 上。

​		使用 RT-Thread Studio 创建的空白工程不包含 LCD 驱动代码，所以务必使用我们提供的模板工程。

### 2. RT-AK 的使用

​		RT-AK 的 gitlab 仓库已经**开源**，可以自行到仓库上克隆到本地 PC。

​		cd 到 RT-AK 的 **rt_ai_tools** 文件夹, 在目录输入 cmd 打开命令行窗口。其中 aitools.py 是核心代码，使用改代码我们可以输入相关命令参数即可快速将 AI 模型 部署至支持的硬件平台上面。

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709144330.png)

​		在 cmd 命令行窗口输入以下命令，参数很好理解，看名字就知道它的意思了。

```bash
python aitools.py --project=C:\Users\Admin\Desktop\Art-Pi_cifar10_with_lcd 
--model=C:\Users\Admin\Desktop\Art-pi\Art-Pi_cifar10\model\cifar10_int8.tflite 
--model_name=cifar10 
--platform=stm32 
--ext_tools=C:\Users\Admin\Desktop\RTAK-tools\stm32ai-windows-5.2.0\windows # x_cube_ai 工具
```

​		运行结果：

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709144405.png)

​		通过这部转换，我们得到的项目工程就集成了 AI 模型, 然后我们可以在上面做相关应用开发，最后编译和烧录到开发板上。

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709144418.png)

我们需要在 **applications** 中书写我们的应用代码，我们给出一个示例 [cifar10_app.c](https://github.com/EdgeAIWithRTT/Project7-Cifar10_Cube_Art-Pi/blob/v0.2.0/Art-Pi_cifar10_with_lcd/applications/cifar10_app.c) ，下面介绍下里面的文件:

```
卷 软件 的文件夹 PATH 列表
卷序列号为 E67E-D1CA
D:.
    cifar10_app.c		// ai 模型推理应用代码实现
    main.c			// artpi LED 闪烁灯例程原 main 函数，未改动
    rt_ai_cifar10_model.c			// 与 STM32 平台相关的模型声明文件
    rt_ai_cifar10_model.h			// 存放 ai 模型输入输出等相关信息文件
    SConscript
```

### 3. cifar10_app.c 核心代码和自定义数据

#### 3. 1 核心代码

​		头文件：

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709144820.png)

​		模型输入数据和 LCD 显示的数据：(两者大小不同，Art-Pi 中的 LCD 尺寸为 320x240, 所以显示的图片尺寸要满足这个大小，而模型输入大小则根据具体模型而言，该项目中的 cifar10 模型输入数据大小为 32x32)

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709144836.png)

​		模型运行核心代码：

```
// cifar10_app.c

...

// 注册模型的代码在 rt_ai_cifar10_model.c 文件下的第43行，代码自动执行
// 模型的相关信息在 rt_ai_cifar10_model.h 文件
// find a registered model handle
model = rt_ai_find(RT_AI_CIFAR10_MODEL_NAME);  // 找到模型
...
result = rt_ai_init(model, work_buffer);  // 初始化模型，传入输入数据
...
result = rt_ai_run(model, ai_run_complete, &ai_run_complete_flag);    // 模型推理一次
...
/* 获取模型输出结果 */
uint8_t *out = (uint8_t *)rt_ai_output(model, 0);
```

#### 3. 2 自定义数据集

​		如何使用自己的图片，生成可以在 Art-Pi 中运行的数据？我们已经提供数据生成代码 [save_img.py](https://github.com/EdgeAIWithRTT/Project7-Cifar10_Cube_Art-Pi/blob/master/cifar10_data/save_img.py)。目录如下

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709145349.png)

在 svae_img.py 文件中修改参数，相关输入会打印到对应 .txt 文件中。然后将 .txt 文件中的信息复制到 cifar10_app.c 文件中，修改参数即可。

### 4. 集成 AI 模型的项目工程的编译和烧录

​		这里我们使用 RT-Thread Studio 进行编译和烧录：

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709145346.png)

### 5. 效果呈现

​		开机屏幕是白色的，

​		当在终端输入 cifar10_app 之后会有 1s 的 logo 显示，然后黑屏。之后终端输出推理结果。

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709145150.png)

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210709145244.png)