<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">

# Deep Learningとは?

### Introduction

*Deep-learning*ネットワークの開発には主に二つのフェーズが必要です。
それが　**学習（training）**　と　**推論（inference）** です。

#### 学習　Training
学習のフェーズでは、ネットワークにラベル付きの大量のデータセットを学ばせます。ニューラルネットワークの重みは学習用のデータセットに含まれるパターンによって認識できるよう最適化されます。Deep neural network はニューロン同士を結合させた多数の層で構成されています。 より深いネットワークでは、訓練と検証により多くの時間がかかりますが、より多くの情報を検出することが可能です。

![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/fd4ba9e7e68b76fc41c8312856c7d0ad)

トレーニング中、ネットワークの推論性能は訓練データセットをつかってテストされて精度向上をしていきます。トレーニングデータセットと同様に、訓練データセットは正解データをラベル付けされており、トレーニングデータセットには含まれていない為ネットワークの精度を評価できます。ネットワークはユーザーによって設定された一定の精度レベルに達するまで、繰り返し訓練を続けます。
Deep Leaningで使うデータセットと深いネットワークサイズのため、通常はトレーニングにコンピュータの計算能力が必要となり、従来のコンピューターアーキテクチャでは数週間から数か月かかる場合があります。しかし、GPUを使用すると、そのプロセスが数日または数週間に至るまで大幅に短縮されます。

##### DIGITS

[DIGITS](https://developer.nvidia.com/digits) を使うとGPUアクセラレーションで誰でも簡単にニューラルネットワークをトレーニングする事ができます。

Using [DIGITS](https://developer.nvidia.com/digits), anyone can easily get started and interactively train their networks with GPU acceleration.  <br />DIGITS is an open-source project contributed by NVIDIA, located here: https://github.com/NVIDIA/DIGITS. 

This tutorial will use DIGITS and Jetson TX1 together for training and deploying deep-learning networks, <br />refered to as the DIGITS workflow:

![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/90bde1f85a952157b914f75a9f8739c2)


#### Inference
Using it's trained weights, the network evaluates live data at runtime.  Called inference, the network predicts and applies reasoning based off the examples it learned.  Due to the depth of deep learning networks, inference requires significant compute resources to process in realtime on imagery and other sensor data.  However, using NVIDIA's GPU Inference Engine which uses Jetson's integrated NVIDIA GPU, inference can be deployed onboard embedded platforms.  Applications in robotics like picking, autonomous navigation, agriculture, and industrial inspection have many uses for deploying deep inference, including:

  - Image recognition
  - Object detection
  - Segmentation 
  - Image registration (homography estimation)
  - Depth from raw stereo
  - Signal analytics
  
