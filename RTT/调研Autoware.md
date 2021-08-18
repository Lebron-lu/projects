[TOC]

# Autoware

>Gitlab: https://gitlab.com/autowarefoundation
>
>Autoware.Auto：https://gitlab.com/autowarefoundation/autoware.auto/AutowareAuto
>
>[Autoware 在 Ubuntu18.04 下的源码安装和配置 ](https://blog.csdn.net/zhao5269/article/details/106827618)

**Autoware 优势：**

- 针对 L4 级别的自动驾驶能力，已具备完备的核心功能
- · **Gitlab 全开放代码**
    · 部署简易，两天上手
    · 丰富的线上学习资源，全球顶尖的开发者社区
- 基于 ROS 开源平台搭建，支持 ROS 开发环境
- **采用 Apache2.0 许可**，支持二次开发，自主封装应用
- 开放的高精度地图工具，实现本地采图、建图、制图

**Autoware 现状：**

​		早期发行了基于 ROS 1 的 Autoware.AI 项目，但由于 ROS 1 的局限性（主要是其机制所造成的无人系统响应时间相对较慢等问题，不适用于高速无人驾驶）, Autoware 已经推出了基于 ROS2 的 [Autoware.Auto](https://www.autoware.auto/) Autoware.Auto 是 Autoware 的彻底重写，应用一流的软件工程实践，包括PR审阅，PR构建，全面的文档，100％代码覆盖率，编码样式指南以及定义的开发和发布过程，所有这些均由开源社区管理。它具有两个不同：

a）为不同的模块（消息和API）定义了清晰的接口

b）为确定性而设计的体系结构，以便可以在实时和开发机器上重现行为。

[autoware-自动驾驶开源套件](https://www.agilex.ai/index/solution/id/5?lang=zh-cn) 提供开源SDK,ROS PACKAGE和使用资源，可以实现车辆线控和节点控制具备路径点录。

# Autoware.Auto 应用开发架构

![image-20210803155017004](https://gitee.com/wonderful4/images/raw/master/imgs/20210803155017.png)

## 1. Autoware.AI 功能

- 3D本地化
- 3D映射
- 路径规划
- 路径跟随
- 加速/制动/转向控制
- 数据记录
- 汽车/行人/物体检测
- 交通信号检测
- 交通灯识别
- 车道检测
- 对象跟踪
- 传感器校准
- 传感器融合
- 面向云的地图
- 连接自动化
- 智能手机导航
- 软件仿真
- 虚拟现实

## 2. Autoware.AI 架构及模块

​		Autoware主要包括 sensing、computing（perception、decision、planning）、actuation等几个部分，如下图所示。

- 其中sensing模块对应的是各类传感器对真实世界中各类数据的采样，例如camera采样图像、LiDAR采样激光点云等，采样数据属于未处理的原始数据，需要输入到computing模块进行计算处理

- computing模块主要是为了对传感器采样的原始数据进行加工处理，最后以为实现安全高效的导航为目的，将规划结果输出给actuation模块。其中computing模块主要分为三个小模块

    - perception（感知模块），这部分要处理localization（通过车辆当前采集传感器数据和已有地图进行自身定位，ps若无地图需要通过SLAM构建地图），然后detection模块负责检测周围与车辆有场景交互的非自身个体（车辆、行人等），prediction模块会对检测初得物体进行未来预测估计，以便提前规划防止碰撞。
    - decision（决策模块），根据之前感知的结果，Autoware决策一个由有限状态机表示的驾驶行为，以便可以选择适当的计划功能。当前的决策方法是基于规则的系统。
    - planning（规划模块），主要是根据决策和起始点和目标点，采用mission和motion模块可以计算出一条kinodynamic的路径

- actuation 模块，表示驱动器模块，如YMC驱动器等，接收planning模块出来的规划结果，经历驱动器实现驱动控制。

    以下框图是 **Autoware.AI 的整体构架**。

    ![image-20210730104724799](https://gitee.com/wonderful4/images/raw/master/imgs/20210803145511.png)

### 2.1 传感 ( Sensing )

​		具体各个模块支持的硬件点击 [link](https://gitlab.com/autowarefoundation/autoware.ai/autoware/-/wikis/Overview)

### 2.2 计算（computing）

#### **2.2.1 感知（Perception）**

​		感知里面包括定位（localization），检测（detection）和预测（prediction）。

a. 定位： 包括激光雷达定位，GNSS定位和航迹推算三部分。

b. 检测：激光雷达，视觉检测，视觉跟踪，融合检测，融合工具，物体跟踪

c. 预测：物体预测，碰撞检测，变道预测

#### **2.2.2 决策（Decision）**

​		决策里面包含决策制定（decision-maker）和状态机（state_machine)两方面。

其中**decision_maker** 通过订阅的感知，地图和当前状态的消息来发布下一个时刻的状态消息。这个状态的变化会激活相应的规划行为。

#### **2.2.3 规划（planning)**

​		这一模块的任务是依据感知和决策模块得到的结果规划全局的和局部（时域）的行为方案。下面是具体的任务分块。

任务级别：路径规划，车道规划，路点规划，路点生成。

行为级别：速度规划，星形规划（astar_planner），网格规划（adas_lattice_planner），路点跟踪。

### 2.3 执行（Actuation)

​		Autoware 最后的输出是一组包含速度，角速度，方向盘转角和曲率的信息。这些信息通过线控的方式发送到车控制器。控制车辆的油门和方向盘。

### 2.4 各个模块都有对应不同的 ROS 1 节点

![img](https://gitee.com/wonderful4/images/raw/master/imgs/20210802144000.png)

## 3. Autoware.Auto 架构

> 详细模块设计介绍参考 [link](https://blog.csdn.net/moyu123456789/article/details/108584169)

![image-20210803151342801](https://gitee.com/wonderful4/images/raw/master/imgs/20210803151342.png)

​		过去的 **Autoware.AI 存在下面两个显著的问题**：

- 没有非常清晰具体的架构设计；

- 存在一些技术问题，比如模块之间存在紧耦合的关系，模块之间功能划分不够明确。

    而设计**新架构的几个目标**：

- 定义一个层次分明的架构；


- 阐述清楚每个模块的角色功能；


- 简化模块之间的接口：

    使得autoware内部之间的处理更加透明；
    模块之间依赖性降低，使得开发人员联合开发更加简单；
    用户可以轻松的使用自己的软件来替换autoware中的模块。

## 4. ROS 2 

> ROS 2 教程：https://www.guyuehome.com/805

​		ROS 是一个用于在不同进程间匿名的发布、订阅、传递信息的中间件。ROS2系统的核心部分是ROS网络(ROS Graph)。ROS网络是指在ROS系统中不同的节点间相互通信的连接关系。

![](https://gitee.com/wonderful4/images/raw/master/imgs/20210730140953.png)

**ROS 2 架构**

![image-20210802174218897](https://gitee.com/wonderful4/images/raw/master/imgs/20210802174218.png)

### 4.1 ROS2 和不同的 DDS 程序

> 原文：http://ros2.bwbot.org/tourial/tourial/new-interface.html

​		上图中的右边是 ROS2 的架构，ROS2 是建立在 DDS 程序的基础上的。 DDS 程序被用来发现节点，序列化和传递信息。[这篇文章](http://design.ros2.org/articles/ros_on_dds.html)详细介绍了DDS程序的开发动机。总而言之，DDS 程序提供了 ROS 系统所需的一些功能，比如分布式发现节点(并不是像 ROS1 那样中心化)，控制传输中的不同的"通信质量（Quality of Service）"选项。

​		**ROS2 支持多种实现方式。**为了能够在ROS2中使用一个DDS实现，需要一个ROS中间件(RMW软件包), 这个包需要利用DDS程序提供的API和工具实现ROS中间件的接口。 为了在ROS2中使用一个DDS实现，有大量的工作需要做。但是为了防止ROS2的代码过于绑定某种DDS程序必须支持至少几种DDS程序。因为用户可能会根据他们的项目需求选择不同的DDS程序。

![image-20210730142220702](https://gitee.com/wonderful4/images/raw/master/imgs/20210730142220.png)

### 4.2 ROS2 DDS 整体架构

[ROS 2 DDS 整体架构](https://hackmd.io/@st9540808/ryteg9D2B#ROS2-DDS-%E6%95%B4%E9%AB%94%E6%9E%B6%E6%A7%8B)

![image-20210802180402931](https://gitee.com/wonderful4/images/raw/master/imgs/20210802180403.png)

## Autoware.IO

​		Autoware的接口项目，将通过专有软件和第三方库以可靠的方式扩展Autoware。 例如包括用于传感器的设备驱动程序，用于车辆线控以及用于SoC板的硬件相关程序。

 	  提供了具有统一的界面设计和测试框架的异构硬件参考平台，支持将成员公司的解决方案集成到支持Autoware.Auto和Autoware.AI软件平台上。

## Autoware 会员

> 原文：http://www.360doc.com/content/20/0309/12/30375878_897907530.shtml

Autoware 基金会分为三个等级，第一级是 Premium，每年会费为5万美元，目前有16家，包括华为/海思，Velodyne，ARM，Linaro 96Boards，LG（提供模拟仿真器），Apex、Autonomoustuff、Kalray（多核加速器初创厂家）、Parkopedia（停车大数据服务公司）、南京润和软件、streetdrone（英国无人车初创公司）、tierIV、itd-lab（双目专家）、esol（日本实时嵌入式系统开发商，全球第三大汽车部件厂家电装持有其20%股份，丰田的视觉系统底层软件平台由其提供，达到ASIL-D级）、MACNICA（日本主要电子产品分销商）、TRI-AD（丰田先进研究院）。Autoware 的线控车由丰田提供，为 Lexus 的LS450H.

其他会员登记如下图所示

![r7k8h4aqz5](https://gitee.com/wonderful4/images/raw/master/imgs/20210809000110.jpeg)

## 其它

支持 Autoware.Auto，PIX 推出 [自动驾驶开发套件 PIXKIT](https://mp.weixin.qq.com/s?__biz=MzUxNTYxMjQzOA==&mid=2247485664&idx=1&sn=7e2229a01b42b7339b42280be0a486e8&chksm=f9b54299cec2cb8fb315fcac1b202ab7555a73f9e127a8a2abff1cf18a82949aba6fe10fd353&scene=21#wechat_redirect)

由 PIX 制作的中文版 Autoware 指导手册，操作视频，内容涵盖从环境部署、软件安装到实车连接、Demo 运行的全套教学、项目移植教学。同时，Autoware 在 Gitlab 有全透明全场景的代码库，丰富的线上学习资源，帮助开发者极速掌握 Autoware，实现多场景实车 Demo。

这是autoware的用户手册 https://github.com/CPFL/Autoware-Manuals
