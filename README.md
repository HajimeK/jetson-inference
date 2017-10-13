<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-header.jpg">

# Deep Learningの導入

ようこそ！Deep Learning 導入のためのトレーニングガイドへ！こちらではNVIDIAより提供されている **[DIGITS](https://github.com/NVIDIA/DIGITS)** と **[Jetson TX1/TX2](http://www.nvidia.com/object/embedded-systems.html)** を活用し、Deep Learningの推論と[deep vision](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/index.html) ランタイムライブラリについてのトレーニングガイドを提供します。

このレポートはNVIDIA **[TensorRT](https://developer.nvidia.com/tensorrt)** を使って、組み込みプラットフォームにニューラルネットワークを効率的に導入し、グラフ最適化、カーネルフュージョン、半精度（FP16)を使って電力効率とパフォーマンスの向上を実現します。

画像認識の為の[`imageNet`](imageNet.h)、オブジェクト検出のための[`detectNet`](detectNet.h) 、セグメンテーションのための[`segNet`](segNet.h)などのビジョンプリミティブは共有の [`tensorNet`](tensorNet.h) オブジェクトから継承されています。ライブカメラからのストリーミングやディスクからの画像処理の例が提供されています。付随するドキュメントについては**[Deep Vision API Reference Specification](https://rawgit.com/dusty-nv/jetson-inference/master/docs/html/index.html)** を参照してください。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/deep-vision-primitives.png" width="800">

> &gt; &nbsp; 最新情報は こちらの投稿**[Parallel ForAll post](https://devblogs.nvidia.com/parallelforall/jetpack-doubles-jetson-inference-perf/)**　をご参照ください。　*JetPack 3.1　による推論性能の低レイテンシーの向上 *. <br/>
> &gt; &nbsp;  イメージセグメンテーションモデルとドローンデータセットによるトレーニングガイドはこちらのリンク **[Image Segmentation](#image-segmentation-with-segnet)** をご参照ください。 <br/>
> &gt; &nbsp; DIGITS & MS-COCO トレーニングデータセットを活用した物体検出のトレーニングガイドはこちら **[Object Detection](#locating-object-coordinates-using-detectnet)** をご参照ください。
> &gt; &nbsp; DIGITS & ImageNet ILSVRC12 datasetを活用した画像認識のトレーニングガイドはこちら **[Image Recognition](#re-training-the-network-with-digits)** をご参照ください。
### **Table of Contents**

* [DIGITS Workflow](#digits-workflow) 
* [System Setup](#system-setup)
* [Building from Source on Jetson](#building-from-source-on-jetson)
* [Classifying Images with ImageNet](#classifying-images-with-imagenet)
	* [Using the Console Program on Jetson](#using-the-console-program-on-jetson)
	* [Running the Live Camera Recognition Demo](#running-the-live-camera-recognition-demo)
	* [Re-training the Network with DIGITS](#re-training-the-network-with-digits)
	* [Downloading Image Recognition Dataset](#downloading-image-recognition-dataset)
	* [Customizing the Object Classes](#customizing-the-object-classes)
	* [Importing Classification Dataset into DIGITS](#importing-classification-dataset-into-digits)
	* [Creating Image Classification Model with DIGITS](#creating-image-classification-model-with-digits)
	* [Testing Classification Model in DIGITS](#testing-classification-model-in-digits)
	* [Downloading Model Snapshot to Jetson](#downloading-model-snapshot-to-jetson)
	* [Loading Custom Models on Jetson](#loading-custom-models-on-jetson)
* [Locating Object Coordinates using DetectNet](#locating-object-coordinates-using-detectnet)
	* [Detection Data Formatting in DIGITS](#detection-data-formatting-in-digits)
	* [Downloading the Detection Dataset](#downloading-the-detection-dataset)
	* [Importing the Detection Dataset into DIGITS](#importing-the-detection-dataset-into-digits)
	* [Creating DetectNet Model with DIGITS](#creating-detectnet-model-with-digits)
	* [Testing DetectNet Model Inference in DIGITS](#testing-detectnet-model-inference-in-digits)
	* [Downloading the Model Snapshot to Jetson](#downloading-the-model-snapshot-to-jetson)
	* [DetectNet Patches for TensorRT](#detectnet-patches-for-tensorrt)
	* [Processing Images from the Command Line on Jetson](#processing-images-from-the-command-line-on-jetson)
	* [Multi-class Object Detection Models](#multi-class-object-detection-models)
	* [Running the Live Camera Detection Demo on Jetson](#running-the-live-camera-detection-demo-on-jetson)
* [Image Segmentation with SegNet](#image-segmentation-with-segnet)
	* [Downloading Aerial Drone Dataset](#downloading-aerial-drone-dataset)
	* [Importing the Aerial Dataset into DIGITS](#importing-the-aerial-dataset-into-digits)
	* [Generating Pretrained FCN-Alexnet](#generating-pretrained-fcn-alexnet)
	* [Training FCN-Alexnet with DIGITS](#training-fcn-alexnet-with-digits)
	* [Testing Inference Model in DIGITS](#testing-inference-model-in-digits)
	* [FCN-Alexnet Patches for TensorRT](#fcn-alexnet-patches-for-tensorrt)
	* [Running Segmentation Models on Jetson](#running-segmentation-models-on-jetson)

**推奨システム環境**

学習環境 GPU:  Maxwell or Pascal-based GPU or AWS P2 instance.  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Ubuntu 14.04 x86_64 or Ubuntu 16.04 x86_64 (see DIGITS [AWS AMI](https://aws.amazon.com/marketplace/pp/B01LZN28VD) image).

推論動作環境:    &nbsp;&nbsp;Jetson TX2 Developer Kit with JetPack 3.0 or newer (Ubuntu 16.04 aarch64).  　
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Jetson TX1 Developer Kit with JetPack 2.3 or newer (Ubuntu 16.04 aarch64).

> **note**:  this [branch](http://github.com/dusty-nv/jetson-inference) is verified against the following BSP versions for Jetson TX1/TX2: <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 3.0 RC <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) inc. TensorRT 3.0 RC <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 3.1 / L4T R28.1 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.0 / L4T R27.1 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3.1 / L4T R24.2.1 aarch64 (Ubuntu 16.04 LTS)

Note ： 本サイトのTensorRT サンプルは Jetson TX1/TX2 モジュールへの使用を前提に記載されておりますが、ホストPCにcuDNNとTensorRTをインストールすることで、ホストPCでコンパイルも可能です。

## DIGITS Workflow

Deep Neural Network (DNNs)と機械学習を体験するのは初めてでしょうか？　そうであれば学習と推論についてこちらの[入門書](docs/deep-learning.md)をご覧ください


<a href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md"><img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/digits-samples.jpg" width="800"></a>

NVIDIAのDeep Learingツールを使って、とても簡単にDNNのトレーニングとハイパフォーマンスなDNNの導入を**開始(https://github.com/NVIDIA/DIGITS/blob/master/docs/GettingStarted.md)** することができます。Tesla等のディスクリートGPUはDIGITSをつかったトレーニングのためサーバーやPCまたはノートブックで使用され、JetsonなどのインテグレートGPUが推論用途として組み込みプラットフォームで活用されます。

<a href="https://github.com/dusty-nv/jetson-inference/blob/master/docs/deep-learning.md"><img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/digits-workflow.jpg" width="700"></a>

NVIDIAの [DIGITS](https://github.com/NVIDIA/DIGITS) は、クラウドやPCの注釈付きデータセットでネットワークモデルを対話的にトレーニングするために使用され、TensorRTとJetsonは、現場で実行する推論を導入するために使用されます。TensorRTは、グラフ最適化と半精度FP16サポートを使用して、2倍以上のDNN推論処理を行います。DIGITSとTensorRTは、高度なAIと知覚を実装できる深いニューラルネットワークの開発と展開のための効果的なワークフローを形成します。

## System Setup

このチュートリアルでは　ホストPC（or AWS) をDNNのトレーニングに使い、推論にJetsonを使います。ホストPCは、最新のJetPackでJetsonをフラッシュする役割もあります。まず、必要なOSとツールを使用してホストPCをセットアップして設定します。

### ホストPCに Ubuntuをインストール

下記よりホストPCにUbuntu 16.04 x86_64をダウンロードしインストールしてください。

```
http://releases.ubuntu.com/16.04/ubuntu-16.04.2-desktop-amd64.iso
http://releases.ubuntu.com/16.04/ubuntu-16.04.2-desktop-amd64.iso.torrent
```

Ubuntu 14.04 x86_64は、apt-getでいくつかのパッケージをインストールする際に、後で修正しても問題ありません。

### ホストPC上で　JetPachの実行

最新の **[JetPack](https://developer.nvidia.com/embedded/jetpack)** をホストPCにダウンロードしてください。最新のBoard Support Package（BSP）でJetsonをフラッシュさせることに加えて、JetPackは自動的にCUDA Toolkitのようなホスト用のツールをインストールします。
機能とインストールされているパッケージのリストについては、JetPack [リリースノート]（https://developer.nvidia.com/embedded/jetpack-notes）　を参照してください。

上記のリンクからJetPackをダウンロードした後、以下のコマンドでホストPCからJetPackを実行してください：

``` bash 
$ cd <directory where you downloaded JetPack>
$ chmod +x JetPack-L4T-3.1-linux-x64.run 
$ ./JetPack-L4T-3.1-linux-x64.run 
```

JetPack GUIが開始されます。**[Install Guide](http://docs.nvidia.com/jetpack-l4t/index.html#developertools/mobile/jetpack/l4t/3.0/jetpack_l4t_install.htm)** に記載の手順に従ってセットアップを完了させて下さい。JetPackは最初、あなたが開発しているJetsonの世代を確認します。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/jetpack-platform.png" width="450">

お使いのJetson TX1もしくはTX2を選択し、`Next`をおして進みます。
次の画面でインストール可能なパッケージのリストが確認できます。ホストにインストールされたパッケージは `Host-Ubuntu`の下の一番上に表示され、Jetsonのパッケージは一番下に表示されます。`Action`カラムをクリックすることで、インストールする個々のパッケージを選択または選択解除することができます。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/jetpack-downloads.png" width="500">

CUDAはDNNのトレーニングにホストで使用されるため、右上のラジオボタンをクリックしてフルインストールを選択することをお勧めします。次に、`Next`を押してセットアップを開始します。JetPackは一連のパッケージをダウンロードしてインストールします。後で必要となった場合でも、すべての.debパッケージが `jetpack_downloads`サブディレクトリの下に格納されています。

ダウンロードが完了したら、JetPackをホストPCからJetsonへインストールする段階に入ります。Jetsonは、devkitに含まれているマイクロUSBポートとケーブルを介してホストPCに接続してください。その後、リカバリー・ボタンを押しながらリセットを押して、Jetsonをリカバリー・モードに入れます。マイクロUSBケーブルを接続してJetsonをリカバリモードにした後にホストPCから `lsusb`と入力すると、NVIDIAデバイスがUSBデバイスのリストの下に表示されます。JetPackは、ホストからのマイクロUSB接続を使用してL4T BSPをJetsonにフラッシュします。

フラッシュ後、Jetsonを再起動し、HDMIディスプレイに接続されていれば、Ubuntuデスクトップが起動します。その後、JetPackはSSH経由でホストからJetsonに接続し、CUDAツールキット、cuDNN、TensorRTのARM aarch64ビルドのように、Jetsonに追加パッケージをインストールします。JetPackがSSH経由でJetsonに接続できるようにするには、ホストPCをイーサネット経由でJetsonにネットワーク接続する必要があります。これは、イーサネットケーブルをホストからJetsonに直接実行するか、または両方のデバイスをルータまたはスイッチに接続することで実現できます。JetPack GUIにて、どのネットワーク接続方法が使用するかを選択する事が可能です。

### ホストPCへNVIDIA PCIe Driverをインストール

ここまでの作業で、JetPackはJetsonを最新のL4T BSPでフラッシュし、JetsonとホストPCの両方にCUDAツールキットをインストールしています。ただし、NVIDIA PCIeドライバは、GPUアクセラレーションによるトレーニングを可能にするためにホストPCにインストールする必要があります。ホストPCから以下のコマンドを実行して、NVIDIAドライバをインストールします

``` bash
$ sudo apt-get install nvidia-375
$ sudo reboot
```
再起動後、NVIDIAドライバは `lsmod`の下に表示されます：

``` bash
$ lsmod | grep nvidia
nvidia_uvm            647168  0
nvidia_drm             49152  1
nvidia_modeset        790528  4 nvidia_drm
nvidia              12144640  60 nvidia_modeset,nvidia_uvm
drm_kms_helper        167936  1 nvidia_drm
drm                   368640  4 nvidia_drm,drm_kms_helper
```

CUDAツールキットとNVIDIAドライバが動作していることを確認するには、CUDAサンプルに付属のテストを実行します。

``` bash
$ cd /usr/local/cuda/samples
$ sudo make
$ cd bin/x86_64/linux/release/
$ ./deviceQuery
$ ./bandwidthTest --memory=pinned
```

### ホストPCへcuDNNをインストール

次のステップは、NVIDIA ** [cuDNN]（https://developer.nvidia.com/cudnn）** ライブラリをホストPCにインストールすることです。
NVIDIAのサイトからlibcudnnパッケージとlibcudnnパッケージをダウンロードしてください。

```
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/Ubuntu16_04_x64/libcudnn6_6.0.20-1+cuda8.0_amd64-deb
https://developer.nvidia.com/compute/machine-learning/cudnn/secure/v6/prod/8.0_20170307/Ubuntu16_04_x64/libcudnn6-dev_6.0.20-1+cuda8.0_amd64-deb
```

次に、以下のコマンドでパッケージをインストールします。

``` bash
$ sudo dpkg -i libcudnn6_6.0.20-1+cuda8.0_amd64.deb
$ sudo dpkg -i libcudnn6-dev_6.0.20-1+cuda8.0_amd64.deb
```

### ホストPCへ NVcaffeをインストール

NVcaffeはNVIDAのGPUに最適化されたCaffeのブランチです。NVcaffeはcuDNNを使用し、DNNのトレーニングのためDIGITSの中で使用されます。それをインストールするには、GitHubからNVcaffeレポをクローンし、ソースからコンパイルします。以下のようなNVcaffe-0.15ブランチを使用してください。

> **note**: このチュートリアルでは、NVcaffeはトレーニング用途としてホストPC上のみで必要です。推論では、TetsorRTをJetsonで使用するため、caffeは必要ありません。

DIGITSで必要とされるPythonを含むCaffeに必要なパッケージがインストールされています。

``` bash
$ sudo apt-get install --no-install-recommends build-essential cmake git gfortran libatlas-base-dev libboost-filesystem-dev libboost-python-dev libboost-system-dev libboost-thread-dev libgflags-dev libgoogle-glog-dev libhdf5-serial-dev libleveldb-dev liblmdb-dev libprotobuf-dev libsnappy-dev protobuf-compiler python-all-dev python-dev python-h5py python-matplotlib python-numpy python-opencv python-pil python-pip python-protobuf python-scipy python-skimage python-sklearn python-setuptools 
$ sudo pip install --upgrade pip
$ git clone -b caffe-0.15 http://github.com/NVIDIA/caffe
$ cd caffe
$ sudo pip install -r python/requirements.txt 
$ mkdir build
$ cd build
$ cmake ../ -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF
$ make --jobs=4
$ make pycaffe
```

Caffeの設定と構築が完了しました。ユーザの〜/ .bashrcを編集して、Caffeツリーへのパスを含めます（以下のパスを自身のパスに編集してください）：

``` bash
export CAFFE_ROOT=/home/dusty/workspace/caffe
export PYTHONPATH=/home/dusty/workspace/caffe/python:$PYTHONPATH
```

変更を有効にするには、ターミナルを閉じてから再度開きます。

### ホストPCへDITISTをインストール

**[DIGITS](https://developer.nvidia.com/digits)** は、対話形式でDNNを訓練し、データセットを管理するPythonベースのWebサービスです。DIGITSワークフローでは、ホストPC上で実行されるため、トレーニングフェーズでネットワークモデルを作成できます。訓練されたモデルは、TensorRTを使用してランタイム推論フェーズのためにホストPCからJetsonにコピーされます。

DIGITSをインストールするには、まず必要なパッケージをインストールし、GitHubからDIGITSリポジトリをクローンします。

``` bash
$ sudo apt-get install --no-install-recommends graphviz python-dev python-flask python-flaskext.wtf python-gevent python-h5py python-numpy python-pil python-pip python-protobuf python-scipy python-tk
$ git clone http://github.com/nvidia/DIGITS
$ cd DIGITS
$ sudo pip install -r requirements.txt
```

#### DIGITS Serverの起動　

ターミナルがDITIGSのディレクトリにあれば、`digits-devserver` Pythonスクリプトを実行しウェブサーバーを起動します。

``` bash
$ ./digits-devserver 
  ___ ___ ___ ___ _____ ___
 |   \_ _/ __|_ _|_   _/ __|
 | |) | | (_ || |  | | \__ \
 |___/___\___|___| |_| |___/ 5.1-dev

2017-04-17 13:19:02 [INFO ] Loaded 0 jobs.
```

DIGITSは `digits / jobs`ディレクトリの下にユーザジョブ（トレーニングデータセットとモデルスナップショット）を保存します。
インタラクティブなDIGITSセッションにアクセスするには、Webブラウザを開き、 `0.0.0.0：5000`にナビゲートしてください。

> **note**:　デフォルトではDIGITSサーバはポート5000から起動しますが、ポートは `--port`引数を` digits-devserver`スクリプトに渡すことで指定できます。

## Building from Source on Jetson

このレポートでは、画像認識のためのライブカメラフィード、ローカライゼーション能力を有する歩行者検出ネットワーク（すなわち、境界ボックスの提供）、およびセグメンテーションのためにGooglenet / Alexnetを実行するためのTensorRT対応ディープ学習プリミティブが提供される。 このレポートは、Jetson上に構築されて実行され、DIGITSサーバーでトレーニングされたホストPCからネットワークモデルを受け入れることを目的としています。

最新のソースは[GitHub]（http://github.com/dusty-nv/jetson-inference）から入手し、Jetson TX1 / TX2でコンパイルできます。

> **note**: 　このブランチ（https://github.com/TKO-mac/jetson-inference）　は、Jetson TX1 / TX2の次のBSPバージョンに対して検証されています。　<br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX2 - JetPack 3.0 / L4T R27.1 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3 / L4T R24.2 aarch64 (Ubuntu 16.04 LTS) <br/>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;> Jetson TX1 - JetPack 2.3.1 / L4T R24.2.1 aarch64 (Ubuntu 16.04 LTS)
      
#### このレポートのクローン

リポジトリを取得するには、Jetsonで選択したフォルダに移動します。まず、gitとcmakeがローカルにインストールされていることを確認します。

``` bash
$ sudo apt-get install git cmake
```

次にこのjetson-inferenceのレポートをクローンします。

``` bash
$ git clone http://github.com/dusty-nv/jetson-inference
```

#### CMakeで設定

cmakeを実行すると、特別なpre-installation スクリプト（CMakePreBuild.sh）が実行され、自動的に依存関係がインストールされます。

``` bash
$ cd jetson-inference
$ mkdir build
$ cd build
$ cmake ../
```

> **note**: cmakeコマンドはCMakePrebuild.shスクリプトを起動し、Jetsonに必要なパッケージがインストールされていることを確認しながらsudoを求めます。スクリプトは、Webサービスからネットワークモデルのスナップショットもダウンロードします。

#### プロジェクトのコンパイル

Step2で作成したjetson-inference / buildディレクトリにて下記を実行してください。

``` bash
$ cd jetson-inference/build			# omit if pwd is already /build from above
$ make
```

アーキテクチャに応じて、パッケージはarmhfまたはaarch64のいずれかにビルドされ、次のディレクトリ構造になります。

```
|-build
   \aarch64		    (64-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
   \armhf           (32-bit)
      \bin			where the sample binaries are built to
      \include		where the headers reside
      \lib			where the libraries are build to
```

バイナリはaarch64 / bin　の中、ヘッダはaarch64 / includeの中、ライブラリはaarch64 / libの中に格納されます。

#### Digging Into the Code

参考として画像認識の為の[`imageNet`](imageNet.h)、オブジェクト検出のための[`detectNet`](detectNet.h) など既に用意されているプリミティブを参照してください。

``` c++
/**
 * Image recognition with GoogleNet/Alexnet or custom models, using TensorRT.
 */
class imageNet : public tensorNet
{
public:
	/**
	 * Network choice enumeration.
	 */
	enum NetworkType
	{
		ALEXNET,
		GOOGLENET
	};

	/**
	 * Load a new network instance
	 */
	static imageNet* Create( NetworkType networkType=GOOGLENET );
	
	/**
	 * Load a new network instance
	 * @param prototxt_path File path to the deployable network prototxt
	 * @param model_path File path to the caffemodel
	 * @param mean_binary File path to the mean value binary proto
	 * @param class_info File path to list of class name labels
	 * @param input Name of the input layer blob.
	 */
	static imageNet* Create( const char* prototxt_path, const char* model_path, const char* mean_binary,
							 const char* class_labels, const char* input="data", const char* output="prob" );

	/**
	 * Determine the maximum likelihood image class.
	 * @param rgba float4 input image in CUDA device memory.
	 * @param width width of the input image in pixels.
	 * @param height height of the input image in pixels.
	 * @param confidence optional pointer to float filled with confidence value.
	 * @returns Index of the maximum class, or -1 on error.
	 */
	int Classify( float* rgba, uint32_t width, uint32_t height, float* confidence=NULL );
};
```

Both inherit from the shared [`tensorNet`](tensorNet.h) object which contains common TensorRT code.

## Classifying Images with ImageNet 

認識、検出、ローカライズやセグメンテーションにすでに利用可能なDeep Learnigのネットワークは複数存在します。
このチュートリアルで強調している最初のDeep Learningは、類似のオブジェクトを識別するために訓練された「imageNet」を使用した**画像認識**です。

[`imageNet`]（imageNet.h）オブジェクトは入力画像を受け取り、各クラスの確率を出力します。 **[1000 objects](data/networks/ilsvrc12_synset_words.txt)**　のImageNetのデータベースで学習されていれば、標準のAlexNetとGoogleNetネットワークは上記の[ステップ2]（＃configuration-with-cmake）の間にダウンロードされます。[`imageNet`]（imageNet.h）を使用する際、[` imagenet-console`]（imagenet-console / imagenet-console.cpp）というコマンドラインインタフェースと、 ` imagenet-camera`]（imagenet-camera / imagenet-camera.cpp）というライブカメラプログラムを提供しています。

### Using the Console Program on Jetson

まず、[`imagenet-console`]（imagenet-console / imagenet-console.cpp）プログラムを使用して、いくつかの例でimageNet認識をテストしてみてください。 画像をロードし、TensorRTと[`imageNet`]（imageNet.h）クラスを使用して推論を行い、分類をオーバーレイして出力画像を保存します。

ビルド後、terminalがaarch64 / binディレクトリにあることを確認してください：

``` bash
$ cd jetson-inference/build/aarch64/bin
```
次に、[`imagenet-console`]（imagenet-console / imagenet-console.cpp）プログラムでサンプル画像を分類します。[`imagenet-console`]（imagenet-console / imagenet-console.cpp）は、入力イメージへのパスと出力イメージへのパス（クラスオーバーレイが印刷されたもの）の2つのコマンドライン引数を受け入れます。

``` bash
$ ./imagenet-console orange_0.jpg output_0.jpg
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-orange.jpg" width="500">

``` bash
$ ./imagenet-console granny_smith_1.jpg output_1.jpg
```
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-apple.jpg" width="500">

次に、[imageNet]（imageNet.h）を使用して、Jetson搭載カメラからのライブビデオ入力を分類します。

### Running the Live Camera Recognition Demo

最後の例と同様に、リアルタイム画像認識デモは/ aarch64 / binにあり、[`imagenet-camera`]（imagenet-camera / imagenet-camera.cpp）と呼ばれています。ライブカメラストリームで動作し、ユーザの引数に応じて、googlenetまたはalexnetにTensorRTをロードします。

``` bash
$ ./imagenet-camera googlenet           # to run using googlenet
$ ./imagenet-camera alexnet             # to run using alexnet
```
1秒あたりのフレーム数（FPS）、ビデオからの分類されたオブジェクト名、および分類されたオブジェクトの信頼度は、OpenGLウィンドウタイトルバーに出力されます。GooglenetとAlexnetは1000クラスのオブジェクトを含むILSVRC12 ImageNetデータベースで訓練されているため、アプリケーションはデフォルトで1000種類までのオブジェクトを認識できます。1000種類のオブジェクトの名前のマッピングは、[data / networks / ilsvrc12_synset_words.txt](http://github.com/dusty-nv/jetson-inference/blob/master/data/networks/ilsvrc12_synset_words.txt)にあるリポジトリに含まれています

> **note**:  デフォルトでは、JetsonのオンボードCSIカメラがビデオソースとして使用されます。オンボードカメラの代わりにUSBウェブカメラを使用する場合は、/ dev / video V4L2デバイスを反映するために、[`imagenet-camera.cpp`]（imagenet-camera / imagenet-camera.cpp）の上部にある `DEFAULT_CAMERA`定義を変更してください。ここで使用したモデルはLogitech C920です。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-orange-camera.jpg" width="800">
<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-apple-camera.jpg" width="800">

### Re-training the Network with DIGITS

上記まででダウロードされた既存のGoogleNetとAlexNetのモデルは、ImageNet ILSVRC12ベンチマークの[1000クラスのオブジェクト]（data / networks / ilsvrc12_synset_words.txt）を使って事前にトレーニングされています。

新しいオブジェクトクラスを認識するために、DIGITSを使用して、新しいデータでネットワークを追加トレーニングすることができます。複数のサブクラスを1つにグループ化するなど、既存のクラスを別々に編成することもできます。たとえば、このチュートリアルでは、1000クラスのうちの230クラスを使用し、それらを12クラスにグループ化し、ネットワークを再テストします。

まず、ILSVRC12イメージをダウンロードして作業を開始するか、** [Image Folder]（https://github.com/NVIDIA/DIGITS/blob/master/docs/ImageFolderFormat.md）** で自分のデータセットを置き換えて開始しましょう。

### Downloading Image Recognition Dataset

画像認識データセットは、分類タイプ（通常はディレクトリ別）でソートされた多数の画像で構成されています。ILSVRC12データセットは、デフォルトのGoogleNetおよびAlexNetモデルのトレーニングに使用されました。サイズは約100GBで、1000種類以上の画像が100万枚含まれています。データセットは、[`imagenet-download.py`]（tools / imagenet-download.py）イメージクローラを使用してDIGITSサーバにダウンロードされます。

データセットをダウンロードするには、まず、DIGITSサーバーに十分なディスク容量（120GB推奨）があることを確認し、データセットが保存されるそのマシンのディレクトリから次のコマンドを実行します。

``` bash
$ wget --no-check-certificate https://nvidia.box.com/shared/static/gzr5iewf5aouhc5exhp3higw6lzhcysj.gz -O ilsvrc12_urls.tar.gz
$ tar -xzvf ilsvrc12_urls.tar.gz
$ wget https://rawgit.com/dusty-nv/jetson-inference/master/tools/imagenet-download.py
$ python imagenet-download.py ilsvrc12_urls.txt . --jobs 100 --retry 3 --sleep 0
```

上記のコマンドでは、スクリプトとともに画像URLのリストがダウンロードされてから、クローラ（Crawler)を起動します。

> **note**: データ量が多いのでしっかりとしたネットワークにつないで実行することをお勧めします。
> 1000 ILSVRC12クラス（100GB）をダウンロードするには、まともな接続でも一定の時間が掛かります。
　　　　　　
クローラは、分類に対応するサブディレクトリにイメージをダウンロードします。各イメージ・クラスは、それ自身のディレクトリに格納され、合計で1000個のディレクトリがあります（ILSVRC12の各クラスに1つ）。フォルダは、次のような命名体系で編成されています。
      
```
n01440764/
n01443537/
n01484850/
n01491361/
n01494475/
...
```
上記のn+8つの数字列はクラスの **synset ID**　となります。クラスの名前文字列は、[`ilsvrc12_synset_words.txt`]（data / networks / ilsvrc12_synset_words.txt）で参照できます。たとえば、synset `n01484850 great white shark`などです。

### Customizing the Object Classes

前のステップでダウンロードしたデータセットは、鳥、植物、果物、魚、犬と猫の品種、車両の種類、魚の種類、魚の種類、魚の種類など、いくつかのコアグループの1000オブジェクトクラスを持つデフォルトのAlexNetとGoogleNetモデルを学習するために使用しました。実際の目的では、オリジナルの1000クラスからなる12個のコアグループを認識するGoogleNetモデルの仲間と考えることができます（例えば、122個の個々の犬種を検出し、すべてを共通の「犬」クラスにまとめます）。これらの12個のコアグループは、1000個の個々のsynsetsを使用するより実用的であり、クラス間で結合することにより、より多くのトレーニングデータとそのグループのより強力な分類が得られます。

DIGITSはフォルダの階層内のデータ入力が必要なため、グループ用のディレクトリを作成し、上記でダウンロードしたILSVRC12のシンセットにシンボリックリンクすることができます。DIGITSは、最上位のグループの下にあるすべてのフォルダの画像を自動的に結合します。ディレクトリ構造は下記のようになります。括弧内の値は、グループを構成するために使用されるクラスの数とリンクされているシンセットIDを示す矢印の横の値を示します。

```
‣ ball/  (7)
	• baseball     (→n02799071)
	• basketball   (→n02802426)
	• soccer ball  (→n04254680)
	• tennis ball  (→n04409515)
	• ...
‣ bear/  (4)
	• brown bear   (→n02132136)
	• black bear   (→n02133161)
	• polar bear   (→n02134084)
	• sloth bear   (→n02134418)
• bike/  (3)
• bird/  (17)
• bottle/ (7)
• cat/  (13)
• dog/  (122)
• fish/   (5)
• fruit/  (12)
• turtle/  (5)
• vehicle/ (14)
• sign/  (2)
```

実際にILSVRC12からリンクされているsynsetはたくさんあるので、私たちは、ディレクトリ構造からデータセットを生成する** [`imagenet-subset.sh`]（tools / imagenet-subset.sh）**　スクリプトを提供します。DIGITSサーバーから次のコマンドを実行します。

``` bash
$ wget https://rawgit.com/dusty-nv/jetson-inference/master/tools/imagenet-subset.sh
$ chmod +x imagenet-subset.sh
$ mkdir 12_classes
$ ./imagenet-subset.sh /opt/datasets/imagenet/ilsvrc12 12_classes
```

この例では、リンクは `12_classes`フォルダに作成され、スクリプトの最初の引数は前の手順でダウンロードしたILSVRC12へのパスです。

### Importing Classification Dataset into DIGITS

ブラウザをDIGITSサーバーインスタンスに移動し、'データセット'タブのドロップダウンから新しい分類データセットを作成することを選択します。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-new-dataset-menu.png" width="250">

`Training Images`パスを前のステップの` 12_classes`フォルダに設定し、次のようにします。

* % for validation:  `10`
* Group Name:  `ImageNet`
* Dataset Name: `ImageNet-ILSVRC12-subset`

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-new-dataset.png)

ページ下部の'Create'ボタンをおして、データセットインポートジョブを起動します。データサブセットのサイズは約20GBなので、サーバーのI / Oパフォーマンスに応じて10〜15分かかります。次に、新しいモデルを作成してトレーニングを開始します。

### Creating Image Classification Model with DIGITS

データインポートジョブが完了したら、DIGITSのホーム画面に戻ります。`Models`タブを選択し、ドロップダウンから新しいClassification Modelを作成することを選択します：

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-new-model-menu.png" width="250">

フォームに以下の設定を行います。

* Select Dataset:  `ImageNet-ILSVRC12-subset`
* Subtract Mean:  `Pixel`
* Standard Networks:  `GoogleNet`
* Group Name:  `ImageNet`
* Model Name:  `GoogleNet-ILSVRC12-subset`

トレーニングするGPUを選択したら、下部にある `Create`ボタンをクリックしてトレーニングを開始します。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-new-model.png)

### Testing Classification Model in DIGITS

学習が30エポックを完了した後、訓練されたモデルは次のように表示されます：

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-model.png)

この時点で、DIGITSのいくつかのサンプル画像について新しいモデルの推論をテストすることができます。上記のプロットと同じページで、`Trained Models`セクションの下にスクロールします。`Test a Single Image`で、試してみたい画像を選択してください（例えば` / ilsvrc12 / n02127052 / n02127052_1203.jpg`）：

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-test-single-image.png" width="350">

`Classify One`　ボタンを押すと、次のようなページが表示されます。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-infer-cat.png)

画像は新しいGoogleNet-12モデルとして'cat'と分類され、元のGoogleNet-1000では'Lynx'に分類されています。これは、LynxカテゴリがGoogleNet-12のcatのトレーニングに含まれていたため、新しいモデルが正常に機能していることを示しています。

### Downloading Model Snapshot to Jetson

学習済みモデルがDIGITSで動作していることを確認したので、Jetsonにモデルスナップショットをダウンロードして抽出しましょう。Jetson TX1 / TX2のブラウザから、DIGITSサーバーと `GoogleNet-ILSVRC12-subset`モデルに移動します。`Trained Models`セクションの下で、ドロップダウンから目的のスナップショット（通常は最高エポックのスナップショット）を選択し、` Download Model`ボタンをクリックします。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-digits-model-download.png" width="650">

また、JetsonとDIGITSサーバーが同じネットワークからアクセスできない場合は、上記の手順を使用してスナップショットを仲介マシンにダウンロードし、SCPまたはUSBスティックを使用してJetsonにコピーすることができます。

次に、下記コマンドでアーカイブを解凍します。

```cd <directory where you downloaded the snapshot>
tar -xzvf 20170524-140310-8c0b_epoch_30.0.tar.gz
```

次に、カスタムスナップショットをTetsorRTにロードし、Jetsonで実行します。

### Loading Custom Models on Jetson

これまで使用していた `imagenet-console`と` imagenet-camera`プログラムは、カスタムモデルのスナップショットを読み込むための拡張コマンドラインパラメータを受け入れます。以下の `$ NET`変数を抽出されたスナップショットのパスに設定します：

``` bash
$ NET=networks/GoogleNet-ILSVRC12-subset

$ ./imagenet-console bird_0.jpg output_0.jpg \
--prototxt=$NET/deploy.prototxt \
--model=$NET/snapshot_iter_184080.caffemodel \
--labels=$NET/labels.txt \
--input_blob=data \
--output_blob=softmax
```

前述のように、分類とコンフィデンスは出力画像にオーバーレイされます。元のネットワークの出力と比較すると、再学習されたGoogleNet-12は、元のGoogleNet-1000と同様の分類を行います。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/imagenet-tensorRT-console-bird.png)

上記の拡張コマンドラインパラメータは、[`imagenet-camera`]（imagenet-camera / imagenet-camera.cpp）を使用してカスタム分類モデルを読み込みます。

## Locating Object Coordinates using DetectNet

上記までの画像認識例は、入力画像全体を表すクラス確率を出力しました。このチュートリアルで紹介している第2のDeep Learningは、オブジェクトの検出と、そのオブジェクトが配置されているビデオの場所の特定（つまり、境界ボックスの抽出）です。これは、 'detectNet'または物体検出/ローカリゼーションネットワークを使用して実行されます。

[`detectNet`]（detectNet.h）オブジェクトは2D画像を入力として受け取り、検出されたバウンディングボックスの座標のリストを出力します。物体検出モデルを学習するために、まず、事前に学習されたImageNet認識モデル（Googlenetのような）が、ソース画像に加えて訓練データセットに含まれる境界座標ラベルと共に使用されます。

チュートリアルには、以下の事前に学習されたDetectNetモデルが含まれています。

1. **ped-100**  (single-class pedestrian detector)
2. **multiped-500**   (multi-class pedestrian + baggage detector)
3. **facenet-120**  (single-class facial recognition detector)
4. **coco-airplane**  (MS COCO airplane class)
5. **coco-bottle**    (MS COCO bottle class)
6. **coco-chair**     (MS COCO chair class)
7. **coco-dog**       (MS COCO dog class)

前の画像認識の例と同様に、detectNetを使用するためのコンソールプログラムとカメラストリーミングプログラムが提供されています。

### Detection Data Formatting in DIGITS

下記を含む物体検出データセットの例です
 [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php) 
 [MS-COCO](http://mscoco.org/)
 その他
 
 KITTIデータセットを使用するには、[KITTIでのDIGITSオブジェクト検出チュートリアル]　(https://github.com/NVIDIA/DIGITS/blob/digits-4.0/digits/extensions/data/objectDetection/README.md)　を参照してください。
 
データセットにかかわらず、DIGITSは検出境界ラベルを取り込むためにKITTIメタデータフォーマットを使用します。これらは、イメージファイル名に対応するフレーム番号を持つテキストファイルで構成されます。内容は次のとおりです。

```
dog 0 0 0 528.63 315.22 569.09 354.18 0 0 0 0 0 0 0
sheep 0 0 0 235.28 300.59 270.52 346.55 0 0 0 0 0 0 0
```

[DIGITS]が使用するフォルダ構造とKITTIラベルのフォーマットについては、下記を参照してください。
[Read more](https://github.com/NVIDIA/DIGITS/blob/digits-4.0/digits/extensions/data/objectDetection/README.md) 

### Downloading the Detection Dataset

 [MS-COCO](http://mscoco.org/) データセットを使用して、カメラのフィード内の日常的なオブジェクトの場所を検出するネットワークをトレーニングし、展開する方法を紹介します。MS-COCOオブジェクトクラスをKITTI形式に変換するには、[`coco2kitti.py`]（tools / coco2kitti.py）スクリプトを参照してください。DIGITSフォルダ構造に入ると、データセットとしてDIGITSにインポートできます。DIGITS / KITTIフォーマットで事前に処理されたMS-COCOのいくつかのクラスの例は、便宜上提供されています。

あなたのDIGITSサーバ上の端末から** DIGITS / KITTI形式の**[sample MS-COCO classes](https://nvidia.box.com/shared/static/tdrvaw3fd2cwst2zu2jsi0u43vzk8ecu.gz)** ダウンロードして抽出してください：

```bash
$ wget --no-check-certificate https://nvidia.box.com/shared/static/tdrvaw3fd2cwst2zu2jsi0u43vzk8ecu.gz -O coco.tar.gz

HTTP request sent, awaiting response... 200 OK
Length: 5140413391 (4.5G) [application/octet-stream]
Saving to: ‘coco.tar.gz’

coco 100%[======================================>]   4.5G  3.33MB/s    in 28m 22s 

2017-04-17 10:41:19 (2.5 MB/s) - ‘coco.tar.gz’ saved [5140413391/5140413391]

$ tar -xzvf coco.tar.gz 
```

飛行機、ボトル、椅子、犬のクラスのDIGITS形式のトレーニングデータが含まれています。[`coco2kitti.py`]（tools / coco2kitti.py）は他のクラスを変換するのに使うことができます。

### Importing the Detection Dataset into DIGITS

ブラウザをDIGITSサーバーインスタンスに移動し、`Datasets`タブのドロップダウンから新しい`Detection Dataset`を作成することを選択します。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-new-dataset-menu.png" width="250">

フォームフィールドで、Aerial dataset を抽出した場所の下にある画像フォルダとラベルフォルダへの次のオプションとパスを指定します。

* Training image folder:  `coco/train/images/dog`
* Training label folder:  `coco/train/labels/dog`
* Validation image folder:  `coco/val/images/dog`
* Validation label folder:  `coco/val/labels/dog`
* Pad image (Width x Height):  `640 x 640`
* Custom classes:  `dontcare, dog`
* Group Name:  `MS-COCO`
* Dataset Name:  `coco-dog`

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-new-dataset-dog.png)

選択したデータセットに名前を付け、ページの下部にある`Create`ボタンをクリックして、インポートを開始します。次に、新しい検出モデルを作成し、それを学習させます。

### Creating DetectNet Model with DIGITS

データインポートが完了したら、DIGITSのホーム画面に戻ります。`Models`タブを選択し、ドロップダウンから新しい`Detection Model` を作成することを選択します。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-new-model-menu.png" width="250">

フォームに次の設定を行います。

* Select Dataset:  `coco-dog`
* Training epochs:  `100`
* Subtract Mean:  `none`
* Solver Type:  `Adam`
* Base learning rate:  `2.5e-05`
* Select `Show advanced learning options`
  * Policy:  `Exponential Decay`
  * Gamma:  `0.99`

#### Selecting DetectNet Batch Size

DetectNetのネットワークのデフォルトバッチサイズ10は、学習中に最大12GBのGPUメモリを消費します。ただし、「Batch Accumulation」フィールドを使用することで、12 GB未満のメモリを搭載したGPUでDetectNetをトレーニングすることもできます。DIGITSサーバーで使用可能なGPUメモリの量に応じて、以下の表を参照してください。


| GPU Memory     | Batch Size                | Batch Accumulation  |
| -------------- |:-------------------------:|:-------------------:|
| 4GB            | 2                         | 5                   |
| 8GB            | 5                         | 2                   |
| 12GB or larger | `[network defaults]` (10) | Leave blank (1)     |

12GB以上のメモリを搭載したカードでトレーニングを行っている場合は、デフォルトで`Batch Size` のままにし、 `Batch Accumulation`は空白のままにしておきます。メモリーの少ないGPUの場合は、上記の設定を使用してください。

#### Specifying the DetectNet Prototxt 

ネットワーク領域で `Custom Network`タブを選択し、[` detectnet.prototxt`]（data / networks / detectnet.prototxt）の内容をコピー/ペーストしてください。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-custom-network.jpg)

DetectNetのprototxtは、リポジトリの[`data / networks / detectnet.prototxt`]（https://github.com/dusty-nv/jetson-inference/blob/master/data/networks/detectnet.prototxt）にあります。

#### Training the Model with Pretrained Googlenet

DetectNetはGooglenetから派生しているため、Googlenetの学習済みのウェイトを使用することを強く推奨します。これは、トレーニングのスピードアップと安定化に役立ちます。Googlenetモデルは [here](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)　からダウンロードするか、またはDIGITSサーバーから次のコマンドを実行してダウンロードしてください。

```bash
wget http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel
```

次に、 `Pretrained Model`フィールドの下でGooglenetへのパスを指定します。

トレーニングするGPUを選択し、モデルの名前とグループを設定します。

* Group Name `MS-COCO`
* Model Name `DetectNet-COCO-Dog`

最後に、下部にある「作成」ボタンをクリックしてトレーニングを開始します。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-new-model-dog.png)

### Testing DetectNet Model Inference in DIGITS

mAP（`Mean Average Precision`）プロットが増加し始めるまで、だいたい50エポック、学習作業をしばらく実行してください。DetectNetの損失関数によってmAPが計算される方法により、mAPのスケールは必ずしも0〜100ではなく、5〜10のmAPでもモデルが機能的であることが示されることに注意してください。使用しているCOCOデータセットの例では、100個のエポックが完了する前に、最新のGPUで数時間のトレーニングを受ける必要があります。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-model-dog.png)

この時点で、DIGITSのいくつかのサンプル画像について新しいモデルの推論をテストすることができます。上記のプロットと同じページで、`Trained Models`セクションの下にスクロールします。　`Visualization Method` に *Bounding Boxes* と `Test a Single Image`の下に試したい画像を選択して設定します（例　/ coco / val / images / dog / 000074.png）。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-visualization-options-dog.png" width="350">

`Test One`ボタンを押すと、次のようなページが表示されます。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-infer-dog.png)


### Downloading the Model Snapshot to Jetson

次に、学習済みのモデルスナップショットをダウンロードしてJetsonに抽出します。あなたのJetson TX1 / TX2のブラウザから、あなたのDIGITSサーバと `DetectNet-COCO-Dog`モデルにナビゲートしてください。`Trained Models`セクションの下で、ドロップダウンから目的のスナップショット（通常は最高エポックのスナップショット）を選択し、` Download Model`ボタンをクリックします。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-digits-model-download-dog.png" width="650">

また、JetsonとDIGITSサーバーが同じネットワークからアクセスできない場合は、上記の手順を使用してスナップショットを仲介マシンにダウンロードし、SCPまたはUSBスティックを使用してJetsonにコピーすることができます。

次に、下記コマンドでアーカイブを解凍します。

```cd <directory where you downloaded the snapshot>
tar -xzvf 20170504-190602-879f_epoch_100.0.tar.gz
```

### DetectNet Patches for TensorRT

オリジナルのDetectNetでは、prototxtにはTensorRTでは利用できないPythonクラスタリング層が存在し、スナップショットに含まれる `deploy.prototxt`から削除する必要があります。このレポでは、[`detectNet`]（detectNet.h）クラスはPythonとは対照的にクラスタリングを処理します。

`deploy.prototxt`の最後に` cluster`という名前の層を削除します：

```
layer {
  name: "cluster"
  type: "Python"
  bottom: "coverage"
  bottom: "bboxes"
  top: "bbox-list"
  python_param {
    module: "caffe.layers.detectnet.clustering"
    layer: "ClusterDetections"
    param_str: "640, 640, 16, 0.6, 2, 0.02, 22, 1"
  }
}
```

このPythonレイヤーを削除して、スナップショットをTensorRTにインポートできるようになりました。

### Processing Images from the Command Line on Jetson

[`detectNet`]（detectNet.h）とTensorRTでテストイメージを処理するには、[` detectnet-console`]（detectnet-console / detectnet-console.cpp）プログラムを使用します。[`detectnet-console`]（detectnet-console / detectnet-console.cpp）は、入力イメージへのパスと出力イメージへのパスを表すコマンドライン引数を受け入れます（バウンディングボックスのオーバーレイをレンダリングします）。いくつかのテスト画像もレポに含まれています。

DIGITSからダウンロードしたモデルを指定するには、下記の `detectnet-console`の構文を使用します。まず、抽出されたスナップショットへのパスを `$ NET`変数に設定します：

``` bash
$ NET=20170504-190602-879f_epoch_100

$ ./detectnet-console dog_0.jpg output_0.jpg \
--prototxt=$NET/deploy.prototxt \
--model=$NET/snapshot_iter_38600.caffemodel \
--input_blob=data \ 
--output_cvg=coverage \
--output_bbox=bboxes
```

> **note:** DetectNetのレイヤ名が上記のデフォルトと一致する場合（つまり、このチュートリアルのプロトタイプを使用している場合）は、 `input_blob`、` output_cvg`、および `output_bbox`の引数を省略することができます。これらのオプションのコマンドラインパラメータは、異なるレイヤ名でカスタマイズされたDetectNetを使用している場合に提供されます。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-dog-0.jpg)

#### Launching With a Pretrained Model

レポに付属している事前に準備されたスナップショットの1つをロードするには、pretrainedモデル名を `detectnet-console`の3番目の引数として指定します：

``` bash
$ ./detectnet-console dog_1.jpg output_1.jpg coco-dog
```

上記のコマンドは、dog_1.jpgを処理し、pre-Trained DetectNet-COCO-Dogモデルを使用してoutput_1.jpgに保存します。これは一種のショートカットです。そうしたくない場合は、モデルがトレーニングを完了するのを待つ必要はありません。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-dog-1.jpg)

#### Pretrained DetectNet Models Available

下記は、本レポート（ `cmake`ステップを実行した後に` data / networks`ディレクトリにある）でダウンロードされたpretrained DetectNetスナップショットのテーブルと学習済みモデルの読み込みに使用される ` detectnet-console`　に関連する引数です。


| DIGITS model            | CLI argument    | classes              |
| ------------------------|-----------------|----------------------|
| DetectNet-COCO-Airplane | `coco-airplane` | airplanes            |
| DetectNet-COCO-Bottle   | `coco-bottle`   | bottles              |
| DetectNet-COCO-Chair    | `coco-chair`    | chairs               |
| DetectNet-COCO-Dog      | `coco-dog`      | dogs                 |
| ped-100                 | `pednet`        | pedestrians          |
| multiped-500            | `multiped`      | pedestrians, luggage |
| facenet-120             | `facenet`       | faces                |

これらはすべて、すでに適用されているPythonのレイヤーパッチも持っています。

#### Running Other MS-COCO Models on Jetson

他のCOCOモデルを試してみましょう。これらのトレーニングデータはすべて上記でダウンロードしたデータセットに含まれています。上記のDIGITSトレーニングの例はcoco-dog modelのものでしたが、サンプルCOCOデータセットに含まれる他のクラスでDetectNetをトレーニングする場合も同じ手順を実行できます。

``` bash
$ ./detectnet-console bottle_0.jpg output_2.jpg coco-bottle
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-bottle-0.jpg)


``` bash
$ ./detectnet-console airplane_0.jpg output_3.jpg coco-airplane
```

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-tensorRT-airplane-0.jpg)

#### Running Pedestrian Models on Jetson

本レポートには、人間を検出するために事前に訓練されたDetectNetモデルも含まれています。`pednet`と` multiped`モデルは歩行者を認識し、 `facenet`は顔を認識します。（[FDDB](http://vis-www.cs.umass.edu/fddb/)）混雑したスペースで複数の人間を同時に検出する例を以下に示します。


``` bash
$ ./detectnet-console peds-007.png output_7.png multiped
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-peds-00.jpg" width="900">

### Multi-class Object Detection Models
歩行者に加えて荷物または手荷物を含む画像に対して乗算モデル（`PEDNET_MULTI`）を使用する場合、第2のオブジェクトクラスは緑色のオーバーレイでレンダリングされます。

``` bash
$ ./detectnet-console peds-008.png output_8.png multiped
```

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/detectnet-peds-01.jpg" width="900">

### Running the Live Camera Detection Demo on Jetson

前の例と同様に、 [`detectnet-camera`](detectnet-camera/detectnet-camera.cpp)は、Jetson搭載カメラからのライブビデオ入力で物体検出ネットワークを実行します。コマンドラインから目的のネットワークのタイプとともに起動します。

``` bash
$ ./detectnet-camera coco-bottle    # detect bottles/soda cans in the camera
$ ./detectnet-camera coco-dog       # detect dogs in the camera
$ ./detectnet-camera multiped       # run using multi-class pedestrian/luggage detector
$ ./detectnet-camera pednet         # run using original single-class pedestrian detector
$ ./detectnet-camera facenet        # run using facial recognition network
$ ./detectnet-camera                # by default, program will run using multiped
```

> **note**:  detectnetの実行中に最大限のパフォーマンスを達成するには、スクリプトを実行してJetsonのクロック制限を上げます。
>  `sudo ~/jetson_clocks.sh`
<br/>

> **note**:  デフォルトでは、JetsonのオンボードCSIカメラがビデオソースとして使用されます。デフォルトでは、JetsonのオンボードCSIカメラがビデオソースとして使用されます。オンボードカメラの代わりにUSBウェブカメラを使用する場合は、/ dev / video V4L2デバイスを反映するために、[`detectnet-camera.cpp`](detectnet-camera/detectnet-camera.cpp) の上部にある DEFAULT_CAMERA定義を変更してください。ここで使用したモデルはLogitech C920です。

<br/>

## Image Segmentation with SegNet

このチュートリアルで紹介している第3のDeep Learningは、画像のセグメンテーションです。セグメンテーションは、画像認識のように画像全体を分類するのではなく、ピクセルレベルで分類が行われる画像認識です。これは、あらかじめ訓練されたイメージネット認識モデル（Alexnetなど）を*畳み込み*することで実現され、ピクセルごとのラベル付けが可能な完全畳み込みセグメンテーションモデルに変わります。環境感知と衝突回避に役立つセグメンテーションは、シーンの前景色や背景など、シーンごとにさまざまな潜在的なオブジェクトのピクセル毎の高密度化をもたらします。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-cityscapes.jpg)

[`segNet`](segNet.h) オブジェクトは2D画像を入力として受け取り、ピクセルごとの分類マスクオーバーレイを持つ第2の画像を出力します。マスクの各ピクセルは、分類されたオブジェクトのクラスに対応します。

> **note**:  セグメント化の詳細については、DIGITS [semantic segmentation](https://github.com/NVIDIA/DIGITS/tree/master/examples/semantic-segmentation) の例を参照してください。

### Downloading Aerial Drone Dataset

画像セグメンテーションの例として、地形を空から撮影するドローンのデータセットを使用します。データセットは、飛行中のドローンの視点をエミュレートし、それが感知している地形に基づいて自動操縦として機能するネットワークを訓練するために、ファーストパーソンビュー（FPV）になっています。

データセットをダウンロードして抽出するには、DIGITSサーバーを実行しているホストPCから次のコマンドを実行します。

``` bash
$ wget --no-check-certificate https://nvidia.box.com/shared/static/ft9cc5yjvrbhkh07wcivu5ji9zola6i1.gz -O NVIDIA-Aerial-Drone-Dataset.tar.gz

HTTP request sent, awaiting response... 200 OK
Length: 7140413391 (6.6G) [application/octet-stream]
Saving to: ‘NVIDIA-Aerial-Drone-Dataset.tar.gz’

NVIDIA-Aerial-Drone-Datase 100%[======================================>]   6.65G  3.33MB/s    in 44m 44s 

2017-04-17 14:11:54 (2.54 MB/s) - ‘NVIDIA-Aerial-Drone-Dataset.tar.gz’ saved [7140413391/7140413391]

$ tar -xzvf NVIDIA-Aerial-Drone-Dataset.tar.gz 
```

このデータセットにはドローンのフライトから捕捉されたさまざまなクリップが含まれていますが、このチュートリアルではFPV / SFWAに焦点を当てています。次に、モデルをトレーニングする前に、DIGITSでトレーニングデータベースを作成します。

### Importing the Aerial Dataset into DIGITS

最初に、ブラウザをDIGITSサーバインスタンスに移動し、データセットタブのドロップダウンから新しいセグメンテーションデータセットを作成することを選択します。

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-create-dataset.png" width="250">

データセット作成フォームで、aerial データセットを抽出した場所の下にある画像フォルダとラベルフォルダへの次のオプションとパスを指定します。

* Feature image folder:  `NVIDIA-Aerial-Drone-Dataset/FPV/SFWA/720p/images`
* Label image folder:   `NVIDIA-Aerial-Drone-Dataset/FPV/SFWA/720p/labels`
* set `% for validation` to 1%
* Class labels:  `NVIDIA-Aerial-Drone-Dataset/FPV/SFWA/fpv-labels.txt`
* Color map:  From text file
* Feature Encoding:  `None`
* Label Encoding:  `None`

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-aerial-dataset-options.png)

選択したデータセットに名前を付け、ページの下部にある`Create`ボタンをクリックして、インポートを開始します。次に、新しいセグメンテーションモデルを作成し、トレーニングを開始します。

### Generating Pretrained FCN-Alexnet

完全畳み込みネットワーク（FCN）Alexnetは、DIGITSおよびTensorRTを使用したセグメンテーションモデルに使用するネットワークトポロジです。畳み込み処理については、この[Parallel ForAll](https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5) の記事を参照してください。DIGITS5の新機能はセグメンテーションデータセットとトレーニングモデルをサポートしています。スクリプトは、AlexnetモデルをFCN-Alexnetに変換するDIGITSセマンティックセグメンテーションの例に含まれています。この基本モデルは、カスタムデータセットの将来のFCN-Alexnetセグメンテーションモデルを学習するための事前学習された出発点として使用されます。

あらかじめ学習されたFCN-Alexnetモデルを生成するには、端末を開き、DIGITSセマンティックセグメンテーションの例に移動し、 `net_surgery`スクリプトを実行します：

``` bash
$ cd DIGITS/examples/semantic-segmentation
$ ./net_surgery.py
Downloading files (this might take a few minutes)...
Downloading https://raw.githubusercontent.com/BVLC/caffe/rc3/models/bvlc_alexnet/deploy.prototxt...
Downloading http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel...
Loading Alexnet model...
...
Saving FCN-Alexnet model to fcn_alexnet.caffemodel
```

### Training FCN-Alexnet with DIGITS

データのインポートが完了したら、DIGITSのホーム画面に戻ります。`Models`タブを選択し、ドロップダウンから新しい`Segmentation Model` を作成することを選択します：

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-create-model.png" width="250">

モデル作成フォームで、上記で作成したデータセットを選択します。`Subtract Mean`をNoneに、` Base Learning Rate`を `0.0001`に設定します。DIGITSでネットワークトポロジを設定するには、 `Custom Network`タブを選択し、` Caffe`サブタブが選択されていることを確認してください。 **[FCN-Alexnet prototxt](https://raw.githubusercontent.com/NVIDIA/DIGITS/master/examples/semantic-segmentation/fcn_alexnet.prototxt)**　をテキストボックスにコピー/ペーストします。最後に `Pretrained Model`を上記の` net_surgery`が生成した出力に設定します： `DIGITS / examples / semantic-segmentation / fcn_alexnet.caffemodel`

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-aerial-model-options.png)

Aerialモデルに名前を付けて、ページの下部にある`Create`ボタンをクリックしてトレーニングを開始します。約5エポック後、「精度」プロット（オレンジ色）は上昇し、モデルは使用可能になります：

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-aerial-model-converge.png)

この時点で、DIGITSのいくつかのサンプル画像について新しいモデルの推論をテストすることができます。

### Testing Inference Model in DIGITS

学習されたモデルをJetsonに転送する前に、まずDIGITSでテストしてみましょう。前のプロットと同じページで、`Trained Models` セクションの下にスクロールします。 `Visualization Model`を*Image Segmentation*　に設定し、` Test a Single Image`の下で試してみる画像を選択してください（例　`/ NVIDIA-Aerial-Drone-Dataset / FPV / SFWA / 720p / images / 0428.png`

<img src="https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-aerial-visualization-options.png" width="350">

`Test One`を押すと、次のような画面が表示されます：

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-digits-aerial-infer.png)

次に、学習されたモデルスナップショットをダウンロードしてJetsonに抽出します。

### FCN-Alexnet Patches for TensorRT

元のFCN-Alexnetには、TensorRTではサポートされていないいくつか不要な層が存在し、スナップショットに含まれる `deploy.prototxt`から削除する必要があります。

`deploy.prototxt`の最後のdeconvとcropのレイヤーを削除します：

```
layer {
  name: "upscore"
  type: "Deconvolution"
  bottom: "score_fr"
  top: "upscore"
  param {
    lr_mult: 0.0
  }
  convolution_param {
    num_output: 21
    bias_term: false
    kernel_size: 63
    group: 21
    stride: 32
    weight_filler {
      type: "bilinear"
    }
  }
}
layer {
  name: "score"
  type: "Crop"
  bottom: "upscore"
  bottom: "data"
  top: "score"
  crop_param {
    axis: 2
    offset: 18
  }
}
```
`deploy.prototxt`の24行目で` pad：100`を `pad：0`に変更します。最後に、aerialのデータセットの `fpv-labels.txt`と` fpv-deploy-colors.txt`をJetsonのモデルスナップショットフォルダにコピーします。FCN-Alexnetモデルのスナップショットは、TensorRTと互換性があります。今我々はJetsonでそれを実行し、画像の推論を行うことができます。

### Running Segmentation Models on Jetson

Jetsonでカスタムセグメンテーションネットワークモデルのスナップショットをテストするには、コマンドラインインターフェイスを使用してsegnet-consoleプログラムをテストします。

まず、抽出したスナップショットのパスを `$ NET`変数に設定します：

``` bash
$ NET=20170421-122956-f7c0_epoch_5.0

$ ./segnet-console drone_0428.png output_0428.png \
--prototxt=$NET/deploy.prototxt \
--model=$NET/snapshot_iter_22610.caffemodel \
--labels=$NET/fpv-labels.txt \
--colors=$NET/fpv-deploy-colors.txt \
--input_blob=data \ 
--output_blob=score_fr
```

これにより、レポと共にダウンロードされたテストイメージ上で、指定されたセグメンテーションモデルが実行されます。

![Alt text](https://github.com/dusty-nv/jetson-inference/raw/master/docs/images/segmentation-aerial-tensorRT.png)

このチュートリアルの事前学習された航空モデルに加え、**[Cityscapes](https://www.cityscapes-dataset.com/)**, **[SYNTHIA](http://synthia-dataset.net/)**, and **[Pascal-VOC](http://host.robots.ox.ac.uk/pascal/VOC/)**　などの他のセグメンテーションデータセットの事前訓練モデルも含まれています。

## Extra Resources

ここでは、Deep Learning開発者のためのリンクが記載されています。

* [Appendix](docs/aux-contents.md)
	* [NVIDIA Deep Learning Institute](https://developer.nvidia.com/deep-learning-institute) — [Introductory QwikLabs](https://developer.nvidia.com/deep-learning-courses)
     * [Building nvcaffe](docs/building-nvcaffe.md)
	* [Other Examples](docs/other-examples.md)
	* [ros_deep_learning](http://www.github.com/dusty-nv/ros_deep_learning) - TensorRT inference ROS nodes

