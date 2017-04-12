import numpy as np
   
def det_ensemble(boxes,ensemble_num, overlapThresh = 0.75):
    results = np.zeros([0,6],float)
 # if there are no boxes, return an empty list
    if len(boxes) == 0:
       return results
 
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes 
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    conf = boxes[:,4]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(conf)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (area[idxs[:last]] - w * h + area[idxs[last]])
        large_ov_idx = np.where(overlap > overlapThresh)[0]
        amount = len(large_ov_idx)
        if not(amount):
            pick.append(i)
            idxs = np.delete(idxs,[last])
        else:
            box_temp = boxes[idxs[large_ov_idx]]
            for j in range(0,len(boxes[i])-1):
                boxes[i,j] = (boxes[i,j] + sum(box_temp[:,j]))
            boxes[i,0:4] /= (amount+1)
            idxs = np.delete(idxs,large_ov_idx)
    results = np.concatenate((results,boxes[pick]),0)
    results[:,4] /= ensemble_num 
    return results