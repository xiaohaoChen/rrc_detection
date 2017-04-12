from __future__ import print_function

import math
import os
import shutil
import stat
import subprocess
import sys

### Modify the following parameters accordingly ###
# The directory which contains the caffe code.
# We assume you are running the script at the CAFFE_ROOT.


caffe_root = './'
os.chdir(caffe_root)
sys.path.insert(0, 'python')

import caffe
from caffe.model_libs import *
from google.protobuf import text_format

# Add extra layers on top of a "base" network (e.g. VGGNet or Inception).
def AddExtraLayers(net, use_batchnorm=True):
    use_relu = True

    # Add additional convolutional layers.
    from_layer = net.keys()[-1]
    ConvBNLayer(net, "conv4_3", "conv4_3r", use_batchnorm, use_relu, 256, 3, 1, 1)
    ConvBNLayer(net, "fc7", "fc7r", use_batchnorm, use_relu, 256, 3, 1, 1)

    # TODO(weiliu89): Construct the name using the last layer to avoid duplication.
    out_layer = "conv6_1"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 1, 0, 1)

    from_layer = out_layer
    out_layer = "conv6_2"
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2)

    for i in xrange(7, 9):
      from_layer = out_layer
      out_layer = "conv{}_1".format(i)
      ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 128, 1, 0, 1)

      from_layer = out_layer
      out_layer = "conv{}_2".format(i)
      ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu, 256, 3, 1, 2)

    return net


rolling_time = 4
branch_num = 4
model_name = "kitti_car"
# Set true if you want to start training right after generating all files.
run_soon = True
# Set true if you want to load from most recently saved snapshot.
# Otherwise, we will load from the pretrain_model defined below.
resume_training = True
# If true, Remove old model files.
remove_old_models = False

# The database file for training data. Created by data/KITTI/create_data.sh
train_data = "data/KITTI-car/lmdb/KITTI-car_training_lmdb"
# The database file for testing data. Created by data/KITTI/create_data.sh
test_data = "data/KITTI-car/lmdb/KITTI-car_testing_lmdb"
# Specify the batch sampler.

resize_width = 2560
resize_height = 768
resize = "{}x{}".format(resize_width, resize_height)

# If true, use batch norm for all newly added layers.
# Currently only the non batch norm version has been tested.
learning_rate = 0.0005
use_batchnorm = False
# Use different initial learning rate.
if use_batchnorm:
    base_lr = 0.04
else:
    # A learning rate for batch_size = 1, num_gpus = 1.
    base_lr = 0.00004
# Modify the job name if you want.
job_name = "RRC_{}_{}".format(resize,model_name)
# The name of the model. Modify it if you want.
model_name = "VGG_KITTI_{}".format(job_name)

# Directory which stores the model .prototxt file.
save_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the snapshot of models.
snapshot_dir = "models/VGGNet/KITTI/{}".format(job_name)
# Directory which stores the job script and log file.
job_dir = "jobs/VGGNet/KITTI/{}".format(job_name)
# model definition files.
train_net_file = "{}/train.prototxt".format(save_dir)
test_net_file = "{}/test.prototxt".format(save_dir)
deploy_net_file = "{}/deploy.prototxt".format(save_dir)
solver_file = "{}/solver.prototxt".format(save_dir)
# snapshot prefix.
snapshot_prefix = "{}/{}".format(snapshot_dir, model_name)
# job script path.
job_file = "{}/{}.sh".format(job_dir, model_name)

# Stores the test image names and sizes. Created by data/KITTI/create_list.sh
name_size_file = "data/KITTI-car/testing_name_size.txt"
# The pretrained model. We use the Fully convolutional reduced (atrous) VGGNet.
pretrain_model = "models/VGGNet/VGG_ILSVRC_16_layers_fc_reduced.caffemodel"
# Stores LabelMapItem.
label_map_file = "data/KITTI-car/labelmap_voc.prototxt"
# L2 normalize conv4_3.
normalizations = [20, -1, -1, -1, -1]
normalizations2 = [-1, -1, -1, -1, -1]
num_outputs=[256,256,256,256,256]
odd=[0,0,0,0,0]
rolling_rate = 0.075
# Solver parameters.
# Defining which GPUs to use.
gpus = "0,1,2,3"
gpulist = gpus.split(",")
num_gpus = len(gpulist)

# Divide the mini-batch to different GPUs.
batch_size = 4 
accum_batch_size = 8
iter_size = accum_batch_size / batch_size
solver_mode = P.Solver.CPU
device_id = 0
batch_size_per_device = batch_size
if num_gpus > 0:
  batch_size_per_device = int(math.ceil(float(batch_size) / num_gpus))
  iter_size = int(math.ceil(float(accum_batch_size) / (batch_size_per_device * num_gpus)))
  solver_mode = P.Solver.GPU
  device_id = int(gpulist[0])

# MultiBoxLoss parameters.
num_classes = 2
share_location = True
background_label_id=0
train_on_diff_gt = True
normalization_mode = P.Loss.VALID
code_type = P.PriorBox.CENTER_SIZE
neg_pos_ratio = 3.
#loc_weight = (neg_pos_ratio + 1.) / 4.
loc_weight = 2.
batch_sampler = [
        {
                'sampler': {
                        },
                'max_trials': 1,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.1,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.3,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.5,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.7,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'min_jaccard_overlap': 0.9,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        {
                'sampler': {
                        'min_scale': 0.3,
                        'max_scale': 1.0,
                        'min_aspect_ratio': 0.5,
                        'max_aspect_ratio': 2.0,
                        },
                'sample_constraint': {
                        'max_jaccard_overlap': 1.0,
                        },
                'max_trials': 50,
                'max_sample': 1,
        },
        ]
train_transform_param = {
        'mirror': True,
        'hsv': True,
        'gaussianblur': True,
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [
                        P.Resize.LINEAR,
                        P.Resize.AREA,
                        P.Resize.NEAREST,
                        P.Resize.CUBIC,
                        P.Resize.LANCZOS4,
                        ],
                },
        'emit_constraint': {
            'emit_type': caffe_pb2.EmitConstraint.CENTER,
            }
        }
test_transform_param = {
        'mean_value': [104, 117, 123],
        'resize_param': {
                'prob': 1,
                'resize_mode': P.Resize.WARP,
                'height': resize_height,
                'width': resize_width,
                'interp_mode': [P.Resize.LINEAR],
                },
        }


multibox_loss_param = {
    'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
    'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
    'loc_weight': loc_weight,
    'num_classes': num_classes,
    'share_location': share_location,
    'match_type': P.MultiBoxLoss.PER_PREDICTION,
    'overlap_threshold': 0.7,
    'use_prior_for_matching': True,
    'background_label_id': background_label_id,
    'use_difficult_gt': train_on_diff_gt,
    'do_neg_mining': True,
    'neg_pos_ratio': neg_pos_ratio,
    'neg_overlap': 0.5,
    'code_type': code_type,
    }
loss_param = {
    'normalization': normalization_mode,
    }

# parameters for generating priors.
# minimum dimension of input image
min_dim = min(resize_width,resize_height) 

mbox_source_layers = ['conv4_3r', 'fc7r', 'conv6_2', 'conv7_2', 'conv8_2']
# in percent %
min_ratio = 15
max_ratio = 85
step = int(math.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - 2)))
min_sizes = []
max_sizes = []
for ratio in xrange(min_ratio, max_ratio + 1, step):
  min_sizes.append(min_dim * ratio / 100.)
  max_sizes.append(min_dim * (ratio + step) / 100.)
min_sizes = [min_dim * 6.7 / 100.] + min_sizes
max_sizes = [[]] + max_sizes
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3]]

# variance used to encode/decode prior bboxes.
if code_type == P.PriorBox.CENTER_SIZE:
  prior_variance = [0.1, 0.1, 0.2, 0.2]
else:
  prior_variance = [0.1]
flip = True
clip = True

# Which layers to freeze (no backward) during training.
freeze_layers = []

# Evaluate on whole test set.
num_test_image = 503
test_batch_size = 1
test_iter = num_test_image / test_batch_size

solver_param = {
    # Train parameters
    'base_lr': learning_rate,
    'weight_decay': 0.0005,
    'lr_policy': "step",
    'stepsize': 30000,
    'gamma': 0.1,
    'momentum': 0.9,
    'iter_size': iter_size,
    'max_iter': 60000,
    'snapshot': 5000,
    'display': 10,
    'average_loss': 10,
    'type': "SGD",
    'solver_mode': solver_mode,
    'device_id': device_id,
    'debug_info': False,
    'snapshot_after_train': True,
    # Test parameters
    'test_iter': [test_iter],
    'test_interval': 1000000,
    'eval_type': "detection",
    'ap_version': "11point",
    'test_initialization': False,
    }

# parameters for generating detection output.

# # Directory which stores the detection results.
# output_result_dir = "results/car/{}/Main".format(job_name)
det_out_param = {
    'num_classes': num_classes,
    'share_location': share_location,
    'background_label_id': background_label_id,
    'nms_param': {'nms_threshold': 0.45, 'top_k': 400},
    # 'save_output_param': {
    #     'output_directory': output_result_dir,
    #     'output_name_prefix': "comp4_det_test_",
    #     'output_format': "VOC",
    #     'label_map_file': label_map_file,
    #     'name_size_file': name_size_file,
    #     'num_test_image': num_test_image,
    #     },
    'keep_top_k': 200,
    'confidence_threshold': 0.001,
    'code_type': code_type,
    }

# parameters for evaluating detection results.
det_eval_param = {
    'num_classes': num_classes,
    'background_label_id': background_label_id,
    'overlap_threshold': 0.5,
    'evaluate_difficult_gt': False,
    'name_size_file': name_size_file,
    }

### Wei Liu said "Hopefully you don't need to change the following" ###
### I do agree with him. ###
# Check file.
check_if_exist(train_data)
check_if_exist(test_data)
check_if_exist(label_map_file)
check_if_exist(pretrain_model)
make_if_not_exist(save_dir)
make_if_not_exist(job_dir)
make_if_not_exist(snapshot_dir)

##########################Create train net.########################################
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
        train=True, output_label=True, label_map_file=label_map_file,
        transform_param=train_transform_param, batch_sampler=batch_sampler)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False, freeze_layers=freeze_layers)

AddExtraLayers(net, use_batchnorm)

mbox_layers = CreateMultiBoxHead_share_2x(net, data_layer='data', from_layers=mbox_source_layers,
        use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
        aspect_ratios=aspect_ratios, normalizations=normalizations,
        num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
        prior_variance=prior_variance, kernel_size=3, pad=1,layers_names=mbox_source_layers,branch_num=branch_num)

# Create the MultiBoxLossLayer.
name = "mbox_loss"
mbox_layers.append(net.label)
net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
        loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
        propagate_down=[True, True, False, False])

for roll_idx in range(1,rolling_time+1):
##################
    roll_layers = CreateRollingStruct(net,from_layers_basename=mbox_source_layers,num_outputs=num_outputs,odd=odd,
        rolling_rate=rolling_rate,roll_idx=roll_idx,conv2=False)

    mbox_layers = CreateMultiBoxHead_share_2x(net, data_layer='data', from_layers=roll_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, normalizations=normalizations2,
            num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1,layers_names=mbox_source_layers, 
            conf_postfix='%d'%(roll_idx+1), loc_postfix='%d'%(roll_idx+1),branch_num=branch_num)
    
    # Create the MultiBoxLossLayer.
    name = "mbox_loss%d"%(roll_idx+1)
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            propagate_down=[True, True, False, False])
#==============================================================================
      
with open(train_net_file, 'w') as f:
    print('name: "{}_train"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

################################################# Create test net.#############################################################
net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file,
        transform_param=test_transform_param)

VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False, freeze_layers=freeze_layers)

AddExtraLayers(net, use_batchnorm)

# create normalizaion layers
for i in range(len(normalizations)):
    if normalizations[i] == -1:
        continue
    from_layer = mbox_source_layers[i]
    norm_name = '%s_norm'%(from_layer)
    net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
data_layer = 'data'
#  calculate the priorbox scales
min_sizes_2x = []
max_sizes_2x = []
for i in range(0,len(min_sizes)-1):
    for j in range(0,branch_num):
        min_sizes_2x.append(min_sizes[i] + j * (min_sizes[i+1] - min_sizes[i])/branch_num)
min_sizes_2x.append(min_sizes[-1])
for i in range(0,len(max_sizes)-1):
    if not(max_sizes[i]):
        for j in range(0,branch_num):
            max_sizes_2x.append([])
    else:
        for j in range(1,branch_num+1):
            max_sizes_2x.append(min_sizes_2x[branch_num*i + j])
max_sizes_2x.append(max_sizes[-1])
#  create multi priorbox layer
priorbox_layers = []
aspect_ratio = []
for i in range(0,len(mbox_source_layers)):
    from_layer = mbox_source_layers[i]
    aspect_ratio = aspect_ratios[i]
    if (i == len(mbox_source_layers) - 1):
        repeat_times = 1
    else:
        repeat_times = branch_num

    for mbox_idx in range(0,repeat_times):
          name = "{}_mbox_priorbox{}".format(from_layer,mbox_idx)
          if max_sizes and max_sizes[i]:
              if aspect_ratio:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx], 
                                        max_size=max_sizes_2x[branch_num*i + mbox_idx],
                                        aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
              else:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx], 
                                        max_size=max_sizes_2x[branch_num*i + mbox_idx],
                                        clip=clip, variance=prior_variance)
          else:
              if aspect_ratio:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx],
                                         aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
              else:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx],
                                         clip=clip, variance=prior_variance)
          priorbox_layers.append(net[name])
name = "mbox_priorbox"
net[name] = L.Concat(*priorbox_layers, axis=2)
#==============================================================================        
for roll_idx in range(1,rolling_time+1):
    roll_layers = CreateRollingStruct(net,from_layers_basename=mbox_source_layers,num_outputs=num_outputs,odd=odd,rolling_rate=rolling_rate,
                                        roll_idx=roll_idx,conv2=False)    
    
#==============================================================================  
mbox_layers = CreateMultiBoxHead_share_2x(net, data_layer='data', from_layers=roll_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, normalizations=normalizations2,
            num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1,layers_names=mbox_source_layers, conf_postfix='%d'%(rolling_time+1), 
            loc_postfix='%d'%(rolling_time+1),branch_num = branch_num) 
conf_name = "mbox_conf%d"%(rolling_time+1)
if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
  reshape_name = "{}_reshape".format(conf_name)
  net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
  softmax_name = "{}_softmax".format(conf_name)
  net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
  flatten_name = "{}_flatten".format(conf_name)
  net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
  mbox_layers[1] = net[flatten_name]
elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
  sigmoid_name = "{}_sigmoid".format(conf_name)
  net[sigmoid_name] = L.Sigmoid(net[conf_name])
  mbox_layers[1] = net[sigmoid_name]

net.detection_out = L.DetectionOutput(*mbox_layers,
    detection_output_param=det_out_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
    detection_evaluate_param=det_eval_param,
    include=dict(phase=caffe_pb2.Phase.Value('TEST')))
with open(test_net_file, 'w') as f:
    print('name: "{}_test"'.format(model_name), file=f)
    print(net.to_proto(), file=f)

#############################Create deploy net.################################

net = caffe.NetSpec()
net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
        train=False, output_label=True, label_map_file=label_map_file,
        transform_param=test_transform_param)
VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
    dropout=False, freeze_layers=freeze_layers)

AddExtraLayers(net, use_batchnorm)

# create normalizaion layers
for i in range(len(normalizations)):
    if normalizations[i] == -1:
        continue
    from_layer = mbox_source_layers[i]
    norm_name = '%s_norm'%(from_layer)
    net[norm_name] = L.Normalize(net[from_layer], scale_filler=dict(type="constant", value=normalizations[i]),
                    across_spatial=False, channel_shared=False)
data_layer = 'data'
#  calculate the priorbox scales
min_sizes_2x = []
max_sizes_2x = []
for i in range(0,len(min_sizes)-1):
    for j in range(0,branch_num):
        min_sizes_2x.append(min_sizes[i] + j * (min_sizes[i+1] - min_sizes[i])/branch_num)
min_sizes_2x.append(min_sizes[-1])
for i in range(0,len(max_sizes)-1):
    if not(max_sizes[i]):
        for j in range(0,branch_num):
            max_sizes_2x.append([])
    else:
        for j in range(1,branch_num+1):
            max_sizes_2x.append(min_sizes_2x[branch_num*i + j])
max_sizes_2x.append(max_sizes[-1])
#  create multi priorbox layer
priorbox_layers = []
aspect_ratio = []
for i in range(0,len(mbox_source_layers)):
    from_layer = mbox_source_layers[i]
    aspect_ratio = aspect_ratios[i]
    if (i == len(mbox_source_layers) - 1):
        repeat_times = 1
    else:
        repeat_times = branch_num

    for mbox_idx in range(0,repeat_times):
          name = "{}_mbox_priorbox{}".format(from_layer,mbox_idx)
          if max_sizes and max_sizes[i]:
              if aspect_ratio:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx], 
                                        max_size=max_sizes_2x[branch_num*i + mbox_idx],
                                        aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
              else:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx], 
                                        max_size=max_sizes_2x[branch_num*i + mbox_idx],
                                        clip=clip, variance=prior_variance)
          else:
              if aspect_ratio:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx],
                                         aspect_ratio=aspect_ratio, flip=flip, clip=clip, variance=prior_variance)
              else:
                  net[name] = L.PriorBox(net[from_layer], net[data_layer], min_size=min_sizes_2x[branch_num*i + mbox_idx],
                                         clip=clip, variance=prior_variance)
          priorbox_layers.append(net[name])
name = "mbox_priorbox"
net[name] = L.Concat(*priorbox_layers, axis=2)
rolling_time = 2
#==============================================================================        
for roll_idx in range(1,rolling_time+1):
    roll_layers = CreateRollingStruct(net,from_layers_basename=mbox_source_layers,num_outputs=num_outputs,odd=odd,rolling_rate=rolling_rate,
                                roll_idx=roll_idx,conv2=False)       
    mbox_layers = CreateMultiBoxHead_share_2x(net, data_layer='data', from_layers=roll_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, normalizations=normalizations2,
            num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1,layers_names=mbox_source_layers, conf_postfix='%d'%(roll_idx+1), 
            loc_postfix='%d'%(roll_idx+1),branch_num=branch_num)   
    conf_name = "mbox_conf%d"%(roll_idx+1)
    if multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.SOFTMAX:
      reshape_name = "{}_reshape".format(conf_name)
      net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
      softmax_name = "{}_softmax".format(conf_name)
      net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
      flatten_name = "{}_flatten".format(conf_name)
      net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
      mbox_layers[1] = net[flatten_name]
    elif multibox_loss_param["conf_loss_type"] == P.MultiBoxLoss.LOGISTIC:
      sigmoid_name = "{}_sigmoid".format(conf_name)
      net[sigmoid_name] = L.Sigmoid(net[conf_name])
      mbox_layers[1] = net[sigmoid_name]
    
    net['detection_out%d'%(roll_idx+1)] = L.DetectionOutput(*mbox_layers,
        detection_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
#==============================================================================
deploy_net = net

with open(deploy_net_file, 'w') as f:
    net_param = deploy_net.to_proto()
    # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
    del net_param.layer[0]    
    net_param.name = '{}_deploy'.format(model_name)
    net_param.input.extend(['data'])
    net_param.input_shape.extend([
        caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
    print(net_param, file=f)

# Create solver.
solver = caffe_pb2.SolverParameter(
        train_net=train_net_file,
        test_net=[test_net_file],
        snapshot_prefix=snapshot_prefix,
        **solver_param)

with open(solver_file, 'w') as f:
    print(solver, file=f)

max_iter = 0
# Find most recent snapshot.
for file in os.listdir(snapshot_dir):
  if file.endswith(".solverstate"):
    basename = os.path.splitext(file)[0]
    iter = int(basename.split("{}_iter_".format(model_name))[1])
    if iter > max_iter:
      max_iter = iter

train_src_param = '--weights="{}" \\\n'.format(pretrain_model)
if resume_training:
  if max_iter > 0:
    train_src_param = '--snapshot="{}_iter_{}.solverstate" \\\n'.format(snapshot_prefix, max_iter)

if remove_old_models:
  # Remove any snapshots smaller than max_iter.
  for file in os.listdir(snapshot_dir):
    if file.endswith(".solverstate"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))
    if file.endswith(".caffemodel"):
      basename = os.path.splitext(file)[0]
      iter = int(basename.split("{}_iter_".format(model_name))[1])
      if max_iter > iter:
        os.remove("{}/{}".format(snapshot_dir, file))

# Create job file.
with open(job_file, 'w') as f:
  f.write('cd {}\n'.format(caffe_root))
  f.write('./build/tools/caffe train \\\n')
  f.write('--solver="{}" \\\n'.format(solver_file))
  f.write(train_src_param)
  if solver_param['solver_mode'] == P.Solver.GPU:
    f.write('--gpu {} 2>&1 | tee {}/{}.log\n'.format(gpus, job_dir, model_name))
  else:
    f.write('2>&1 | tee {}/{}.log\n'.format(job_dir, model_name))

# Run the job.
os.chmod(job_file, stat.S_IRWXU)
if run_soon:
  subprocess.call(job_file, shell=True)

