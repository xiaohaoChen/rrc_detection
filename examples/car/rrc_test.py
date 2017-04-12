import numpy as np
#%matplotlib inline
import timeit
import Image
import ImageDraw 

# Make sure that the work directory is caffe_root
caffe_root = './' 
# modify img_dir to your path of testing images of kitti
img_dir = '/your/path/to/KITTI/testing/image_2/'
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')
from google.protobuf import text_format
from caffe.proto import caffe_pb2

import caffe
from _ensemble import *
caffe.set_device(0)
caffe.set_mode_gpu()
num_img =7518
model_def = 'models/VGGNet/KITTI/RRC_2560x768_kitti_car/deploy.prototxt'
model_weights = 'models/VGGNet/KITTI/RRC_2560x768_kitti_car/VGG_KITTI_RRC_2560x768_kitti_car_iter_60000.caffemodel'
voc_labelmap_file = caffe_root+'data/KITTI-car/labelmap_voc.prototxt'
save_dir = 'models/VGGNet/KITTI/RRC_2560x768_kitti_car/result-test/'
txt_dir = 'models/VGGNet/KITTI/RRC_2560x768_kitti_car/result-test/'

detection_out_num = 3 
if not(os.path.exists(txt_dir)):
    os.makedirs(txt_dir)
if not(os.path.exists(save_dir)):
    os.makedirs(save_dir)    
file = open(voc_labelmap_file, 'r')
voc_labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), voc_labelmap) 
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
image_width = 2560 
image_height = 768

net.blobs['data'].reshape(1,3,image_height,image_width)
    
def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames
 
    
for img_idx in range(0,num_img):
    det_total = np.zeros([0,6],float)
    ensemble_num = 0
    img_file = img_dir+'{:06d}.png'.format(img_idx)
    print 'processing image {:06d}.png\n'.format(img_idx)
    image = caffe.io.load_image(img_file)
    
    transformed_image = transformer.preprocess('data', image)
    net.blobs['data'].data[...] = transformed_image
    
    # t1 = timeit.Timer("net.forward()","from __main__ import net")
    # print t1.timeit(2)
    # Forward pass.
    net_out = net.forward()
    for out_i in range(2,detection_out_num + 1):
        detections = net_out['detection_out%d'%(out_i)].copy()
       
    # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]
   # Get detections with confidence higher than 0.001
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.001]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(voc_labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]* image.shape[1]
        top_ymin = det_ymin[top_indices]* image.shape[0]
        top_xmax = det_xmax[top_indices]* image.shape[1]
        top_ymax = det_ymax[top_indices]* image.shape[0]

        det_this = np.concatenate((top_xmin.reshape(-1,1),top_ymin.reshape(-1,1),
                                   top_xmax.reshape(-1,1),top_ymax.reshape(-1,1),
                                   top_conf.reshape(-1,1),det_label[top_indices].reshape(-1,1)),1)

        ensemble_num = ensemble_num + 1
        det_total = np.concatenate((det_total,det_this),0)
#   evaluate the flipped image
    image_flip = image[:,::-1,:]
    transformed_image = transformer.preprocess('data', image_flip)
    net.blobs['data'].data[...] = transformed_image
    net_out = net.forward()
    for out_i in range(2,detection_out_num + 1):
        detections = net_out['detection_out%d'%(out_i)].copy()
        temp = detections[0,0,:,3].copy()
        detections[0,0,:,3] = 1-detections[0,0,:,5]
        detections[0,0,:,5] = 1-temp

        # Parse the outputs.
        det_label = detections[0,0,:,1]
        det_conf = detections[0,0,:,2]
        det_xmin = detections[0,0,:,3]
        det_ymin = detections[0,0,:,4]
        det_xmax = detections[0,0,:,5]
        det_ymax = detections[0,0,:,6]

        # Get detections with confidence higher than 0.1.
        top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.0]

        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_labels = get_labelname(voc_labelmap, top_label_indices)
        top_xmin = det_xmin[top_indices]* image.shape[1]
        top_ymin = det_ymin[top_indices]* image.shape[0]
        top_xmax = det_xmax[top_indices]* image.shape[1]
        top_ymax = det_ymax[top_indices]* image.shape[0]

        det_this = np.concatenate((top_xmin.reshape(-1,1),top_ymin.reshape(-1,1),
                                   top_xmax.reshape(-1,1),top_ymax.reshape(-1,1),
                                   top_conf.reshape(-1,1),det_label[top_indices].reshape(-1,1)),1)
        ensemble_num = ensemble_num + 1
        det_total = np.concatenate((det_total,det_this),0)

    #ensemble different outputs

    det_results = det_ensemble(det_total,ensemble_num)
    idxs = np.where(det_results[:,4] > 0.0001)[0]
    top_xmin = det_results[idxs,0]
    top_ymin = det_results[idxs,1]
    top_xmax = det_results[idxs,2]
    top_ymax = det_results[idxs,3]
    top_conf = det_results[idxs,4]
    top_label = det_results[idxs,5]
    result_file = open(save_dir+"%06d.txt"%(img_idx),'w')
    # img = Image.open(img_dir + "%06d.png"%(img_idx))
    # draw = ImageDraw.Draw(img)       
    for i in xrange(top_conf.shape[0]):
        xmin = top_xmin[i]
        ymin = top_ymin[i]
        xmax = top_xmax[i]
        ymax = top_ymax[i]
        h = float(ymax - ymin)
        w = float(xmax - xmin)
        if (w==0) or (h==0):
           continue
        if (h/w >=2)and((xmin<10)or(xmax > 1230)):
           continue
        score = top_conf[i]
        label = 'Car'   
        # if score > 0.1:
        #     draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(0,255,0))
        #     draw.text((xmin,ymin),'%.2f'%(score),fill=(255,255,255))
        # elif score > 0.02:
        #     draw.line(((xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)),fill=(255,0,255))
        #     draw.text((xmin,ymin),'%.2f'%(score),fill=(255,255,255))
        result_file.write("%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1000 -1000 -1000 -10 %.8f\n"%(label,xmin,ymin,xmax,ymax,score))
        # img.save(save_dir+"%06d.png"%(img_idx))
