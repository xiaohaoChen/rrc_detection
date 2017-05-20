# Accurate Single Stage Detector Using Recurrent Rolling Convolution
By [Jimmy Ren](http://www.jimmyren.com/), Xiaohao Chen, Jianbo Liu, Wenxiu Sun, Jiahao Pang, Qiong Yan, Yu-Wing Tai, Li Xu.

### Introduction

High localization accuracy is crucial in many real-world applications. We propose a novel
single stage end-to-end object detection network (RRC) to produce high accuracy detection results. You can use the code to train/evaluate a network for object detection task. For more details, please refer to our paper (https://arxiv.org/abs/1704.05776).

| method | KITTI test *mAP* car (moderate)|
| :-------: | :-----: |
| [Mono3D](http://3dimage.ee.tsinghua.edu.cn/cxz/mono3d)| 88.66% |
| [SDP+RPN](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Yang_Exploit_All_the_CVPR_2016_paper.pdf)| 88.85% |
| [MS-CNN](https://github.com/zhaoweicai/mscnn) | 89.02% |
| [Sub-CNN](https://arxiv.org/pdf/1604.04693.pdf) | 89.04% |
| RRC (single model) | **89.85%** |  

[KITTI ranking](http://www.jimmyren.com/papers/rrc_kitti.pdf)

### Citing RRC

Please cite RRC in your publications if it helps your research:

@inproceedings{Ren17CVPR,    
  author = {Jimmy Ren and Xiaohao Chen and Jianbo Liu and Wenxiu Sun and Jiahao Pang and Qiong Yan and Yu-Wing Tai and Li Xu},       
  title = {Accurate Single Stage Detector Using Recurrent Rolling Convolution},      
  booktitle = {CVPR},         
  year = {2017}     
}
### Contents
1. [Installation](#installation)
2. [Preparation](#preparation)
3. [Train/Eval](#traineval)
4. [Models](#models)
4. [Ackonwledge](#Acknowledge)
### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$CAFFE_ROOT`
   ```Shell
   https://github.com/xiaohaoChen/rrc_detection.git
   cd rrc_detection
   ```
2. Build the code. Please follow [Caffe instruction](http://caffe.berkeleyvision.org/installation.html) to install all necessary packages and build it.
   Before build it, you should install CUDA and CUDNN(v5.0).    
   CUDA 7.5 and CUDNN v5.0 were adapted in our computer.
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
   Unzip the training images, testing images and the labels in `$HOME/data/KITTI/`.

3. Create the LMDB file.
   For training .
   As only the images contain cars are adopted as training set for car detection,  the labels for cars should be extracted.      
   We have provided the list of images contain cars in `$CAFFE_ROOT/data/KITTI-car/`.
   ```Shell
   # extract the labels for cars
   cd $CAFFE_ROOT/data/KITTI-car/
   ./extract_car_label.sh
   ```

   Before create the LMDB files. The labels should be converted to VOC type. We provide some matlab scripts.     
   The scripts are in `$CAFFE_ROOT/data/convert_labels/`. Just modify `converlabels.m`.
   ```Shell
   line 4: root_dir = '/your/path/to/KITTI/';
   ```
   VOC type labels will be generated in `$KITTI_ROOT/training/labels_2car/xml/`.
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
1. Train your model and evaluate the model.
   ```Shell
   # It will create model definition files and save snapshot models in:
   #   - $CAFFE_ROOT/models/VGGNet/KITTI/RRC_2560x768_kitti_car/
   # and job file, log file in:
   #   - $CAFFE_ROOT/jobs/VGGNet/KITIIT/RRC_2560x768_kitti_car/
   # After 60k iterations, we can get the model as we said in the paper (mAP 89.*% in KITTI).
   python examples/car/rrc_kitti_car.py
   # Before run the testing script. You should modify [line 10: img_dir] to [your path to kitti testing images].
   python examples/car/rrc_test.py
   ```
   We train our models in a computer with 4 TITAN X(Maxwell) GPU cards. By default, we assume you train the models on mechines with 4 TITAN X GPUs.       
   If you only have one TITAN X card, you should modify the script `rrc_kitti.py`.    
   ```Shell
   line 118: gpus = "0,1,2,3" -> gpus = "0"
   line 123: batch_size = 4   -> batch_size = 1
   ```
   If you have two TITAN X cards, you should modify the script `rrc_kitti.py` as follow.
   ```Shell
   line 118: gpus = "0,1,2,3" -> gpus = "0,1"
   line 123: batch_size = 4   -> batch_size = 2
   ```
   You can submit the result at [kitti submit](http://www.cvlibs.net/datasets/kitti/user_login.php).
   If you don't have time to train your model, you can download a pre-trained model from the link as follow.    
   [Google Drive](https://drive.google.com/open?id=0ByGD7RFf_dTxS2ZWcWo5cTVQaDQ)    
   [Baidu Cloud](https://pan.baidu.com/s/1c2H0NxY)    
   Unzip the files in `$caffe_root/models/VGGNet/KITTI/`, and run the testing script `rrc_test.py`, you will get the same result as the single model result we showed in the  paper.
   ```Shell
   # before run the script, you should modify the kitti_root at line 10.
   # Make sure that the work directory is caffe_root
   cd $caffe_root
   python models/VGGNet/KITTI/RRC_2560x768_kitti_4r4b_max_size/rrc_test.py
   ```
2. Evaluate the most recent snapshot.
   For testing a model you trained, you show modify the path in `rrc_test.py`.

### Acknowledge
Thanks to Wei Liu, we have benifited a lot from his previous work [SSD (Single Shot Multibox Detector)](https://arxiv.org/abs/1512.02325) and his [code](https://github.com/weiliu89/caffe/tree/ssd).
