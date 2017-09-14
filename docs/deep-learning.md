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
DIGITSはNVIDIAが提供するオープンソースプロジェクトで、こちらのHPより入手できます。 https://github.com/NVIDIA/DIGITS.

このチュートリアルでは、DIGITSワークフローと呼ばれるディープ学習ネットワークのトレーニングとトレーニングのために、DIGITSとJetson TX1を一緒に使用します。

![Alt text](https://a70ad2d16996820e6285-3c315462976343d903d5b3a03b69072d.ssl.cf2.rackcdn.com/90bde1f85a952157b914f75a9f8739c2)


#### 推論　（Inference）
訓練された重みによって、ネットワークは実行時に入力データを判定します。　推論（Inference）と呼ばれるこのネットワークの処理は、学習した例に基づい推論を出力します。Deep Learning ネットワークが深いため、推論は画像および他のセンサデータ上でリアルタイムで処理するために、かなりの計算リソースを必要とします。しかし、Jetsonに搭載されたNVIDIA GPUを使用するGPU推論エンジンを活用して、推論をオンボードの組み込みプラットフォームに導入することができます。
自律航法、農業、産業検査などのロボット工学のアプリケーションには、組み込みプラットフォーム（Jetson)に導入された推論を活用する用途があります。

  - 画像認識　（Image recognition）
  - 物体検出　（Object detection）
  - セグメンテーション　（Segmentation） 
  - 画像登録　（Image registration (homography estimation)）
  - ステレオカメラからの深さ認識　（Depth from raw stereo）
  - 信号解析　（Signal analytics）
  
