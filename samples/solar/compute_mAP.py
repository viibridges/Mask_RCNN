import numpy as np
from ipdb import set_trace as st

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
'''

def yx2xy(yx_bboxes):
    """ convert yx bboxes (Mask RCNN outputs) to xy bboxes """
    xy_bboxes = yx_bboxes[:,[1,0,3,2]]
    return xy_bboxes


"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre



def compute_mAP(pred_classes, pred_bboxes, pred_scores, gt_classes, gt_bboxes, verbose=False):
    APs = compute_ap(pred_classes, pred_bboxes, pred_scores, gt_classes, gt_bboxes, verbose)
    return sum(APs.values()) / len(APs)


def compute_ap(pred_classes, pred_bboxes, pred_scores, gt_classes, gt_bboxes, verbose=False):
    """ Computer average precision for each class.
        Inputs:
            pred_classes: [[class_name/class_id] * #bboxes] * #images. Element could be a numpy array or a list of string
            pred_bboxes: [[x1,y1,x2,y2] * #bboxes] * #images. Element could be a list of 2D numpy array of 4 columns
            pred_scores: [[confidence] * #bboxes] * #images. Element could be a list or 1D numpy array
            gt_classes: [[class_name/class_id] * #bboxes]] * #images. Element could be a numpy array or a list of string
            gt_bboxes: [[x1,y1,x2,y2] *#bboxes] * #images. Element could be a 2D numpy array of 4 columns
        Output:
            APs: A dictionary ({'class_name/class_id': the average precision of this class}
    """

    MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge)

    ## handle ground truths
    gths = []
    gt_counter_per_class = {}
    counter_images_per_class = {}
    for gt_class_id, gt_bbox in zip(gt_classes, gt_bboxes):
        gg = []
        already_seen_classes = []
        for class_id, bbox in zip(gt_class_id, gt_bbox):
            if class_id in gt_counter_per_class:
                gt_counter_per_class[class_id] += 1
            else:
                gt_counter_per_class[class_id] = 1

            if class_id not in already_seen_classes:
                if class_id in counter_images_per_class:
                    counter_images_per_class[class_id] += 1
                else:
                    # if class didn't exist yet
                    counter_images_per_class[class_id] = 1
                already_seen_classes.append(class_id)

            gg.append({'class': class_id, 'bbox': bbox, 'used': False})
        gths.append(gg)


    ## handle predictions
    preds = []
    for pred_class_id, pred_bbox, pred_score in zip(pred_classes, pred_bboxes, pred_scores):
        pp = []
        for class_id, bbox, score in zip(pred_class_id, pred_bbox, pred_score):
            pp.append({'class': class_id, 'bbox': bbox, 'score': score})
        preds.append(pp)

    # collect bboxes per class (for predictions only)
    bboxes = {}
    for image_id, pred in enumerate(preds):
        for bb in pred:
            if bb['class'] not in bboxes:
                bboxes[bb['class']] = []
            bboxes[bb['class']].append({'bbox': bb['bbox'], 'score': bb['score'], 'id': image_id})

    # sort predictions by score
    for key in bboxes.keys():
        bboxes[key].sort(key=lambda x: x['score'])


    all_pred_classes = set(bboxes.keys())
    all_gt_classes = set(gt_counter_per_class.keys())
    assert all_pred_classes.issubset(all_gt_classes), \
            "In original code, mAP is computed as sum(AP[pred_class]) / #gt_classes. Make sure your test set covers all detection cases."

    APs = {}
    count_true_positives = {}
    for class_id in bboxes.keys():
        count_true_positives[class_id] = 0
        dr_data = bboxes[class_id]

        """
         Assign detection-results to ground-truth objects
        """
        nd = len(dr_data)
        tp = [0] * nd # creates an array of zeros of size nd
        fp = [0] * nd
        for idx, detection in enumerate(dr_data):
            ground_truth_data = gths[detection['id']]
            ovmax = -1
            gt_match = -1
            # load detected object bounding-box
            bb = detection["bbox"]
            for obj in ground_truth_data:
                # look for a class match
                if obj["class"] == class_id:
                    bbgt = obj["bbox"]
                    bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                    iw = bi[2] - bi[0] + 1
                    ih = bi[3] - bi[1] + 1
                    if iw > 0 and ih > 0:
                        # compute overlap (IoU) = area of intersection / area of union
                        ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                        + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                        ov = iw * ih / ua
                        if ov > ovmax:
                            ovmax = ov
                            gt_match = obj

            # set minimum overlap
            min_overlap = MINOVERLAP
            if ovmax >= min_overlap:
                if not bool(gt_match["used"]):
                    # true positive
                    tp[idx] = 1
                    gt_match["used"] = True
                    count_true_positives[class_id] += 1
                else:
                    # false positive (multiple detection)
                    fp[idx] = 1
            else:
                # false positive
                fp[idx] = 1
                if ovmax > 0:
                    status = "INSUFFICIENT OVERLAP"

        #print(tp)
        # compute precision/recall
        cumsum = 0
        for idx, val in enumerate(fp):
            fp[idx] += cumsum
            cumsum += val
        cumsum = 0
        for idx, val in enumerate(tp):
            tp[idx] += cumsum
            cumsum += val
        #print(tp)
        rec = tp[:]
        for idx, val in enumerate(tp):
            rec[idx] = float(tp[idx]) / gt_counter_per_class[class_id]
        #print(rec)
        prec = tp[:]
        for idx, val in enumerate(tp):
            prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
        #print(prec)

        ap, mrec, mprec = voc_ap(rec[:], prec[:])
        APs[class_id] = ap
        if verbose:
            print("class '{}' AP: {}%".format(class_id, ap*100))

    return APs




if __name__ == '__main__':

    pred_classes = [
        ['apple', 42],
        [42]
    ]
    pred_bboxes = [
        [[0,0,100,100], [85, 12, 120, 55]], # or np.array([[0,0,100,100], [85, 12, 120, 55]])
        [[12, 920, 1010, 999]]
    ]
    pred_scores = [
        [.12, .88],
        [42]
    ]
    gt_classes = [
        [42],
        [42, 'apple']
    ]
    gt_bboxes = [
        [[5,10,99,105]],
        [[12, 920, 1010, 999], [12, 920, 1010, 999]]
    ]

    mAP = compute_mAP(pred_classes, pred_bboxes, pred_scores, gt_classes, gt_bboxes, verbose=True)
    print(mAP)
