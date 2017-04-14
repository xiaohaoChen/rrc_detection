%%%convert the kitti type labels to voc type labels
currentFolder = pwd;
addpath(genpath(currentFolder));
root_dir = '/your/path/to/KITTI/';
img_type = '.png';
img_dir = [root_dir 'training/image_2/'];
txt_dir = [root_dir 'training/label_2car/'];
xml_dir = [root_dir 'training/label_2car/xml/'];

if ~exist(xml_dir,'dir')
    mkdir(xml_dir);
end

file_type = '.txt';

shuffle = 0;  

file_names = dir(txt_dir);
total = length(file_names);

if shuffle
    shuffle_idx = randperm(total);
    file_names = file_names(shuffle_idx);
end

cls_count = zeros(2,1);
for i = 1:total 
    if file_names(i).isdir
        continue;
    end
    if mod(i,100) == 0
        fprintf('%d images are processed\n',i)
    end
    file_name = file_names(i).name;
    [path, name, ext] = fileparts(file_name);
    if exist([xml_dir name '.xml'],'file')
       continue 
    end
    if ~strcmp(file_type, ext) 
        continue
    end       
    txt_path = [txt_dir '/' file_name];
    label_idx = int32(str2num(name));
    objects = readLabels(txt_dir,label_idx);
    
    obj_voc = [];
    
    obj_voc.annotation.folder = 'training/image_2/';
    obj_voc.annotation.filename = [name img_type];
    img = imread([img_dir name img_type]);
    obj_voc.annotation.size.width = num2str(size(img,2));
    obj_voc.annotation.size.height = num2str(size(img,1));   
    obj_voc.annotation.size.depth = num2str(size(img,3));
    for o = 1:numel(objects)
       if ~(strcmp(objects(o).type,'Car'))
           continue
       end
       obj_voc.annotation.object(o).name = objects(o).type;
       obj_voc.annotation.object(o).truncated = num2str(objects(o).truncation);
       obj_voc.annotation.object(o).occluded = num2str(objects(o).occlusion);
       obj_voc.annotation.object(o).alpha = num2str(objects(o).alpha);
       obj_voc.annotation.object(o).bndbox.xmin = num2str(objects(o).x1);
       obj_voc.annotation.object(o).bndbox.xmax = num2str(objects(o).x2);
       obj_voc.annotation.object(o).bndbox.ymin = num2str(objects(o).y1);
       obj_voc.annotation.object(o).bndbox.ymax = num2str(objects(o).y2);
       obj_voc.annotation.object(o).dimensions.height = num2str(objects(o).h);
       obj_voc.annotation.object(o).dimensions.width = num2str(objects(o).w);
       obj_voc.annotation.object(o).dimensions.length = num2str(objects(o).l);
       obj_voc.annotation.object(o).location.x = num2str(objects(o).t(1));
       obj_voc.annotation.object(o).location.y = num2str(objects(o).t(2));
       obj_voc.annotation.object(o).location.z = num2str(objects(o).t(3));
       obj_voc.annotation.object(o).rotation_y = num2str(objects(o).ry);
    end
    VOCwritexml(obj_voc,[xml_dir name '.xml']);
end
