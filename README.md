# Accurate Single Stage Detector Using Recurrent Rolling Convolution


### Introduction

You can use the code to train/evaluate a network for object detection task. For more details, please refer to our paper (TBA).

| method | KITTI test *mAP* car(moderate)|
| ------- | ----- |
| [MS-CNN](https://github.com/zhaoweicai/mscnn) | 89.02% | 
| [Sub-CNN](https://arxiv.org/pdf/1604.04693.pdf)| 89.04% | 
| [SDP+RPN](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Exploit_All_the_CVPR_2016_paper.pdf)| 88.85% |
| [Mono3D](http://3dimage.ee.tsinghua.edu.cn/cxz/mono3d)| 88.66% |
| RRC (single model) | **89.85%** |
| RRC (ensemble) | **90.19%** |


### Citing RRC
TBA


### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)
4. [Contact](#contact)

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
   ```Shell
   https://github.com/xiaohaoChen/rrc_detection.git
   cd rrc_detection
   ```

2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
   Before build it, you should install CUDA and CUDNN(v5.0)
   ```Shell
   # Modify Makefile.config according to your Caffe installation.
   cp Makefile.config.example Makefile.config
   make -j8
   # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
   make py
   make test -j8
   make runtest -j8
   ```

### Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). 
   By default, we assume the model is stored in `$CAFFE_ROOT/models/VGGNet/`.

2. Download the KITTI dataset(http://www.cvlibs.net/datasets/kitti/eval_object.php). 
   By default, we assume the data is stored in `$HOME/data/KITTI/`
   uzip the training images, testing images and the labels in `$HOME/data/KITTI/`.

3. Create the LMDB file.
   For training the models for car detection.
   As only the images contain cars are adopted for training set, we should extract the labels for cars from the original labels.
   We have provided the list of images contain cars in `$CAFFE_ROOT/data/KITTI-car/`.
   ```Shell
   # extract the labels for cars
   cd $CAFFE_ROOT/data/KITTI-car/
   ./extrach_car_label.sh
   ```
   Before create the LMDB files. The labels should be converted to VOC type. We provide some matlab scripts to finish this. The scripts are in `$CAFFE_ROOT/data/convert_labels/`.\<br> 
   Just modify line 4 in converlabels.m (`root_dir = '/your/path/to/KITTI/';`) to your path to kitti, and run the script. VOC type labels will be generated in `$KITTI_ROOT/training/labels_2car/xml/`. 
   ```Shell
   cd $CAFFE_ROOT/data/KITTI-car/
   # Create the trainval.txt, test.txt, and test_name_size.txt in data/KITTI-car/
   ./create_list.sh
   # You can modify the parameters in create_data.sh if needed.
   # It will create lmdb files for trainval and test with encoded original image:
   #   - $HOME/data/KITTI/lmdb/KITTI-car_training_lmdb/
   #   - $HOME/data/KITTI/lmdb/KITTI-car_testing_lmdb/
   # and make soft links at data/KITTI-car/lmdb
    ./data/KITTI-car/create_data.sh
   ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
   ```Shell
   # It will create model definition files and save snapshot models in:
   #   - $CAFFE_ROOT/models/VGGNet/KITTI/RRC_2560x768_kitti_car/
   # and job file, log file in:
   #   - $CAFFE_ROOT/jobs/VGGNet/KITIIT/RRC_2560x768_kitti_car/
   # After 60k iterations, we can get the model as we said in the paper (mAP 89.*% in KITTI).
   python examples/car/ssd_kitti_car.py
   # Before run the testing script. You should modify [line 10: img_dir] to [your path to kitti testing images].
   python examples/car/rrc_test.py
   ```
   You can submit the result at [kitti submit](http://www.cvlibs.net/datasets/kitti/user_login.php).
   If you don't have time to train your model, you can download a pre-trained model at [here](). 
   Unzip our files to `$caffe_root/models/VGGNet/KITTI/`, and run the testing script, you will get the same result as the single model result we showed in the  paper.
   ```Shell
   # before run the script, you should modify the kitti_root at line 10.
   cd $caffe_root
   python models/VGGNet/KITTI/RRC_2560x768_kitti_4r4b_max_size/rrc_test.py
   ```

2. Evaluate the most recent snapshot.
   For testing a model you trained, you show modify the path in rrc_test.py.


### ACKNOWLEDGE
Thanks to Wei Liu, we have benifited a lot from his previous work [SSD (Single Shot Multibox Detector)](https://github.com/weiliu89/caffe/tree/ssd).

