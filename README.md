# YAD2K: Yet Another Darknet 2 Keras

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

## Welcome to YAD2K

You only look once, but you reimplement neural nets over and over again.

YAD2K is a 90% Keras/10% Tensorflow implementation of YOLO_v2.

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.

![YOLO_v2 COCO model with test_yolo defaults](etc/dog_small.jpg)

--------------------------------------------------------------------------------

## Requirements

- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [h5py](http://www.h5py.org/) (For Keras model serialization.)
- [Pillow](https://pillow.readthedocs.io/) (For rendering test results.)
- [Python 3](https://www.python.org/)
- [pydot-ng](https://github.com/pydot/pydot-ng) (Optional for plotting model.)

### 安裝
```bash
git clone https://github.com/allanzelener/yad2k.git
cd yad2k

# [Option 1] To replicate the conda environment:
conda env create -f environment.yml
source activate yad2k
# [Option 2] Install everything globaly.
pip install numpy h5py pillow
pip install tensorflow-gpu  # CPU-only: conda install -c conda-forge tensorflow
pip install keras # Possibly older release: conda install keras
```

## 快速開始

- 從[YOLO官方網站](http://pjreddie.com/darknet/yolo/)下載Darknet模型的設置檔與權重檔到專案根目錄。例如:使用MS COCO資料集訓練的預訓練模型
    - 下載[YOLOv2 608x608 設置檔(yolo.cfg)](https://github.com/pjreddie/darknet/blob/master/cfg/yolo.cfg)
	- 下載[YOLOv2 608x608 權重檔(yolo.weights)](https://pjreddie.com/media/files/yolo.weights)
- 將原本使用Darknet預訓練的YOLO_v2模型透過`yad2k.python`命令稿來轉換為Keras模型
    - 例如: `python yad2k.py yolo.cfg yolo.weights model_data/yolov2_coco_608x608.h5`
- 把一些圖像複製到`images/`的子目錄來測試轉換後Keras的YOLOv2模型。
    - 例如: `python test_yolo.py model_data/yolov2_coco_608x608.h5`，偵測的結果會置放在`images/out/`的目錄裡

完整範例如下:

```bash
wget http://pjreddie.com/media/files/yolo.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
./yad2k.py yolo.cfg yolo.weights model_data/yolov2_coco_608x608.h5
./test_yolo.py model_data/yolov2_coco_608x608.h5  # output in images/out/
```

請參閱 `./yad2k.py --help`和`./test_yolo.py --help`以獲取更多設定選項。

--------------------------------------------------------------------------------

## 入門 (Getting Started)
* [demo.ipynb](/demo.ipynb) 是最簡單開始學習使用YOLO的方法。它展示了一個使用MS COCO預先訓練的模型來偵測圖像中的物體的範例。
它包括在任意圖像上運行物體偵測和為每個偵測到的物體產生邊界框(bounding box)的程式碼。

* [train_shapes.ipynb](train_shapes.ipynb) shows how to train Mask R-CNN on your own dataset. This notebook introduces a toy dataset (Shapes) to demonstrate training on a new dataset.

* ([model.py](model.py), [utils.py](utils.py), [config.py](config.py)): These files contain the main Mask RCNN implementation. 


* [inspect_data.ipynb](/inspect_data.ipynb). This notebook visualizes the different pre-processing steps
to prepare the training data.

* [inspect_model.ipynb](/inspect_model.ipynb) This notebook goes in depth into the steps performed to detect and segment objects. It provides visualizations of every step of the pipeline.

* [inspect_weights.ipynb](/inspect_weights.ipynb)
This notebooks inspects the weights of a trained model and looks for anomalies and odd patterns.

## More Details

The YAD2K converter currently only supports YOLO_v2 style models, this include the following configurations: `darknet19_448`, `tiny-yolo-voc`, `yolo-voc`, and `yolo`.

`yad2k.py -p` will produce a plot of the generated Keras model. For example see [yolo.png](etc/yolo.png).

YAD2K assumes the Keras backend is Tensorflow. In particular for YOLO_v2 models with a passthrough layer, YAD2K uses `tf.space_to_depth` to implement the passthrough layer. The evaluation script also directly uses Tensorflow tensors and uses `tf.non_max_suppression` for the final output.

`voc_conversion_scripts` contains two scripts for converting the Pascal VOC image dataset with XML annotations to either HDF5 or TFRecords format for easier training with Keras or Tensorflow.

`yad2k/models` contains reference implementations of Darknet-19 and YOLO_v2.

`train_overfit` is a sample training script that overfits a YOLO_v2 model to a single image from the Pascal VOC dataset.

## Known Issues and TODOs

- Expand sample training script to train YOLO_v2 reference model on full dataset.
- Support for additional Darknet layer types.
- Tuck away the Tensorflow dependencies with Keras wrappers where possible.
- YOLO_v2 model does not support fully convolutional mode. Current implementation assumes 1:1 aspect ratio images.

## Darknets of Yore

YAD2K stands on the shoulders of giants.

- :fire: [Darknet](https://github.com/pjreddie/darknet) :fire:
- [Darknet.Keras](https://github.com/sunshineatnoon/Darknet.keras) - The original D2K for YOLO_v1.
- [Darkflow](https://github.com/thtrieu/darkflow) - Darknet directly to Tensorflow.
- [caffe-yolo](https://github.com/xingwangsfu/caffe-yolo) - YOLO_v1 to Caffe.
- [yolo2-pytorch](https://github.com/longcw/yolo2-pytorch) - YOLO_v2 in PyTorch.

--------------------------------------------------------------------------------
