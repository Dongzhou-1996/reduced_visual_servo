import numpy as np
import cv2
import sys
from time import time
from sot_3d_loader import SOT_3D_Loader
import kcftracker
import time

import kcftracker

selectingObject = False
initTracking = False
onTracking = False
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 1
duration = 0.01

if __name__ == '__main__':
    tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
    sot_dataset = SOT_3D_Loader('F:\\Object_Tracking\\zed-master\\dataset\\train_dataset\\satellite', 'test')
    cv2.namedWindow('visualization', cv2.WINDOW_GUI_EXPANDED)
    for s, sequence in enumerate(sot_dataset):
        print("sequence: %d" % s)
        print("sequence length: %d" % sequence.__len__())
        print("sequence path: %s" % sot_dataset.sequence_directories[s])
        initialization = 1
        for f, (left_img_file, left_roi, right_img_file, right_roi) in enumerate(sequence):
            print("--frame: %d" % f)
            left_img = cv2.imread(left_img_file, 1)
            right_img = cv2.imread(right_img_file, 1)
            if initialization == 1:
                tracker.init([left_roi[0], left_roi[1], left_roi[2], left_roi[3]], left_img)
                initialization = 0
                continue
            else:
                t0 = time.time()
                boundingbox = tracker.update(left_img)
                t1 = time.time()

                boundingbox = list(map(int, boundingbox))
                cv2.rectangle(left_img, (boundingbox[0], boundingbox[1]),
                              (boundingbox[0] + boundingbox[2], boundingbox[1] + boundingbox[3]),
                              color=(0, 255, 0), thickness=2)

                duration = 0.8 * duration + 0.2 * (t1 - t0)
                # duration = t1-t0
                cv2.putText(left_img, 'FPS: ' + str(1 / duration)[:4].strip('.'), (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

            left_pt1 = [left_roi[0], left_roi[1]]
            left_pt2 = [left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]]
            # right_pt1 = [right_roi[0], right_roi[1]]
            # right_pt2 = [right_roi[0] + right_roi[2], right_roi[1] + right_roi[3]]
            cv2.rectangle(left_img, pt1=tuple(left_pt1), pt2=tuple(left_pt2),
                          color=(255, 0, 255), thickness=2)
            # cv2.rectangle(right_img, pt1=tuple(right_pt1), pt2=tuple(right_pt2),
            # 			  color=(255, 0, 255), thickness=2)
            # visualization = np.hstack((left_img, right_img))

            cv2.imshow('visualization', left_img)
            c = cv2.waitKey(1) & 0xFF
            if c == 27:
                exit(1)
            elif c == ord('n'):
                break

