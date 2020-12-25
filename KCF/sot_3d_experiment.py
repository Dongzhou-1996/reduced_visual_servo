from __future__ import absolute_import, division, print_function

import os
import numpy as np
import glob
import ast
import json
import time
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import cv2
import argparse
from .sot_3d_loader import SOT_3D_Loader
from .kcftracker import KCFTracker
from got10k.utils.metrics import rect_iou
from got10k.utils.ioutils import compress

class Experimet3DSOT(object):
    """Experiment pipeline and evaluation toolkit for 3D_SOT dataset.
        Args:
            root_dir (string): Root directory of 3D_SOT dataset where
                ``train``, ``val`` and ``test`` folders exist.
            subset (string): Specify ``train``, ``val`` or ``test``
                subset of 3D_SOT.
            list_file (string, optional): If provided, only run experiments on
                sequences specified by this file.
            result_dir (string, optional): Directory for storing tracking
                results. Default is ``./results``.
            report_dir (string, optional): Directory for storing performance
                evaluation results. Default is ``./reports``.
    """

    def __init__(self, root_dir, subset='val', result_dir='results',
                 report_dir='reports', use_dataset=True):
        super(Experimet3DSOT, self).__init__()
        assert subset in ['val', 'test']
        self.subset = subset
        if use_dataset:
            self.dataset = SOT_3D_Loader(root_dir, subset=subset)
        self.result_dir = os.path.join(result_dir, 'SOT_3D')
        self.report_dir = os.path.join(report_dir, 'SOT_3D')
        self.nbins_iou = 201
        self.repetitions = 3

    def run(self, tracker, save_video=False, overwrite_result=False):
        print('Running tracker %s on SOT3D...' % tracker.name)
        # loop over the complete dataset
        for s, sequence in enumerate(self.dataset):
            sequence_name = self.dataset.sequence_directories[s].split('\\')[-1]
            print('--Sequence %s (%dth in total %d)' % (sequence_name,
                s + 1, len(self.dataset)))
            print('--Sequence length %d' % sequence.__len__())
            # run multiple repetitions for each sequence
            for r in range(self.repetitions):
                # check if the tracker is deterministic
                if r > 0 and tracker.is_deterministic:
                    break
                elif r == 3 and self._check_deterministic(
                        tracker.name, sequence_name):
                    print('  Detected a deterministic tracker, ' +
                          'skipping remaining trials.')
                    break

                # skip if results exist
                print(' Repetition: %d' % (r + 1))
                record_file = os.path.join(
                    self.result_dir, tracker.name, sequence_name,
                    '%s_%03d_bbox.txt' % (sequence_name, r + 1))
                if os.path.exists(record_file) and not overwrite_result:
                    print('Found results, skipping', sequence_name)
                else:
                    # tracking loop
                    print("=> SOT3D tracker staring ...")
                    tracking_results = []
                    times = []
                    for f, (left_img_file, left_roi, right_img_file, right_roi) in enumerate(sequence):
                        print('--Frame %dth in total %d' % (f, sequence.__len__()))
                        left_img = cv2.imread(left_img_file, 1)
                        right_img = cv2.imread(right_img_file, 1)
                        if left_img is None or right_img is None:
                            print("frame {} failed!".format(f))
                            print("=> skipping to track next sequence ...")
                            break
                        if f == 0:
                            print("=> SOT3D tracker initializing ...")
                            tracker.init(left_roi, left_img)
                            print("=> The initialization of SOT3D tracker have been done!")
                        else:
                            start_time = time.time()
                            bbox_2d = tracker.update(left_img)

                            # print("bbox2d: {}".format(bbox2d))
                            end_time = time.time()
                            runtime = end_time - start_time
                            bbox_2d = list(map(int, bbox_2d))
                            tracking_results.append(bbox_2d)
                            times.append(runtime)

                    # record results
                    tracking_results = np.array(tracking_results).reshape(-1, 4)
                    print(tracking_results)
                    self._record(record_file, tracking_results, times)

                # save videos
                if save_video:
                    video_file = record_file[:record_file.rfind('_')] + '_video.avi'
                    width, height = eval(self.dataset.meta['resolution'])
                    out_video = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'MJPG'), 30, (width, height), True)
                    tracking_results = np.loadtxt(record_file, dtype=float, delimiter=',').astype(int)
                    for f, (left_img_file, left_roi_gt, right_img_file, right_roi_gt) in enumerate(sequence):
                        print('--Frame %dth in total %d' % (f, sequence.__len__()))
                        if f == 0:
                            # skip the first frame which is used to initialize tracker
                            continue
                        left_img = cv2.imread(left_img_file, 1)

                        # bbox2d ground-truth
                        left_pt1_gt = [left_roi_gt[0], left_roi_gt[1]]
                        left_pt2_gt = [left_roi_gt[0] + left_roi_gt[2], left_roi_gt[1] + left_roi_gt[3]]

                        # bbox2d result of tracker
                        left_roi = tracking_results[f - 1, :4]
                        left_pt1 = [left_roi[0], left_roi[1]]
                        left_pt2 = [left_roi[0] + left_roi[2], left_roi[1] + left_roi[3]]

                        # draw groud-truth bbox2d
                        cv2.rectangle(left_img, pt1=tuple(left_pt1_gt), pt2=tuple(left_pt2_gt),
                                      color=(255, 255, 0), thickness=2)

                        # draw bbox2d of tracker
                        cv2.rectangle(left_img, pt1=tuple(left_pt1), pt2=tuple(left_pt2),
                                      color=(0, 255, 0), thickness=2)

                        cv2.namedWindow('visualization', cv2.WINDOW_GUI_EXPANDED)
                        cv2.imshow('visualization', left_img)
                        key = cv2.waitKey(1)
                        if key & 0xFF == 27:
                            print('=> tracking process stopped !')
                            exit(1)
                        elif key & 0xFF == ord('n'):
                            print("=> skipping to track next sequence ...")
                            break
                        out_video.write(left_img)
                    out_video.release()
                    print('  Videos saved at', video_file)


    def report(self, tracker_names, plot_curves=True):
        assert isinstance(tracker_names, (list, tuple))

        # if self.subset == 'test':
        #     pwd = os.getcwd()
        #
        #     # generate compressed submission file for each tracker
        #     for tracker_name in tracker_names:
        #         # compress all tracking results
        #         result_dir = os.path.join(self.result_dir, tracker_name)
        #         os.chdir(result_dir)
        #         save_file = '../%s' % tracker_name
        #         compress('.', save_file)
        #         print('Records saved at', save_file + '.zip')
        #
        #     # switch back to previous working directory
        #     os.chdir(pwd)
        #
        #     return None
        # elif self.subset == 'val':
        # assume tracker_names[0] is your tracker

        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        report_file = os.path.join(report_dir, 'performance.json')

        performance = {}
        for name in tracker_names:
            print('=> {} evaluating'.format(name))
            left_ious = {}
            times = {}
            performance.update({name: {
                'sequence_wise': {}}})

            for s, sequence in enumerate(self.dataset):
                sequence_name = self.dataset.sequence_directories[s].split('\\')[-1]
                print('--Sequence %s (%dth in total %d)' % (sequence_name,
                                                            s + 1, len(self.dataset)))
                record_files = glob.glob(os.path.join(
                    self.result_dir, name, sequence_name,
                    '*_bbox.txt'))
                if len(record_files) == 0:
                    continue
                    raise Exception('Results for sequence %s not found.' % sequence_name)

                left_annotations = np.array(sequence.left_annotation).reshape(-1, 4)
                right_annotations = np.array(sequence.right_annotation).reshape(-1, 4)
                # read results of all repetitions
                left_boxes = []

                for record_file in record_files:
                    data = np.loadtxt(record_file, dtype=float, delimiter=',')
                    if (data.shape[0] == left_annotations.shape[0] - 1) and (data.shape[0] == right_annotations.shape[0] - 1):
                        left_boxes.append(data[:, :4])
                    else:
                        print("the dimension of tracking results is not matched to ground-truth")
                        continue

                # calculate and stack all ious
                bound = ast.literal_eval(self.dataset.meta['resolution'])
                left_sequence_ious = [rect_iou(b[0:], left_annotations[1:], bound=bound) for b in left_boxes]
                left_sequence_ious = np.concatenate(left_sequence_ious)
                left_ious[os.path.join(sequence_name, 'left')] = left_sequence_ious

                # stack all tracking times
                times[sequence_name] = []
                time_files = glob.glob(os.path.join(
                    self.result_dir, name, sequence_name,
                    '*_time.txt'))
                time_data = [np.loadtxt(t, delimiter=',') for t in time_files]
                sequence_times = np.concatenate(time_data)
                if len(sequence_times) > 0:
                    times[sequence_name] = sequence_times

                # store sequence-wise performance
                left_ao, left_sr, left_speed, _ = self._evaluate(left_sequence_ious, sequence_times)

                performance[name]['sequence_wise'].update(
                    {os.path.join(sequence_name, 'left'): {
                    'ao': left_ao,
                    'sr': left_sr,
                    'speed_fps': left_speed,
                    'length': sequence.__len__() - 1}})


            left_ious = np.concatenate(list(left_ious.values()))
            times = np.concatenate(list(times.values()))

            # store overall performance
            left_ao, left_sr, left_speed, left_succ_curve = self._evaluate(left_ious, times)
            performance[name].update({'overall_left': {
                'ao': left_ao,
                'sr': left_sr,
                'speed_fps': left_speed,
                'succ_curve': left_succ_curve.tolist()}})

        # save performance
        with open(report_file, 'w') as f:
            json.dump(performance, f, indent=4)
        # plot success curves
        if plot_curves:
            keys = ['overall_left']
            self._plot_curves([report_file], tracker_names[0], keys)
        return performance

    def _check_deterministic(self, tracker_name, sequence_name):
        record_dir = os.path.join(
            self.result_dir, tracker_name, sequence_name)
        record_files = sorted(glob.glob(os.path.join(
            record_dir, '%s_[0-9]*.txt' % sequence_name)))

        if len(record_files) < 3:
            return False

        records = []
        for record_file in record_files:
            with open(record_file, 'r') as f:
                records.append(f.read())

        return len(set(records)) == 1

    def _record(self, record_file, boxes, times):
        # record bounding boxes
        record_dir = os.path.dirname(record_file)
        if not os.path.isdir(record_dir):
            os.makedirs(record_dir)
        np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        while not os.path.exists(record_file):
            print('warning: recording failed, retrying...')
            np.savetxt(record_file, boxes, fmt='%.3f', delimiter=',')
        print('Results recorded at', record_file)

        # record running times
        time_file = record_file[:record_file.rfind('_')] + '_time.txt'
        np.savetxt(time_file, times, fmt='%.8f', delimiter=',')

    def _evaluate(self, ious, times):
        # AO, SR and tracking speed
        ao = np.mean(ious)
        sr = np.mean(ious > 0.5)
        if len(times) > 0:
            # times has to be an array of positive values
            speed_fps = np.mean(1. / times)
        else:
            speed_fps = -1

        # success curve
        # thr_iou = np.linspace(0, 1, 101)
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        bin_iou = np.array([[i >= thr for thr in thr_iou] for i in ious])
        # bin_iou = np.greater(ious[:, None], thr_iou[None, :])
        succ_curve = np.mean(bin_iou, axis=0)
        return ao, sr, speed_fps, succ_curve


    def _plot_curves(self, report_files, tracker_name, keys, extension='.png'):
        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_name)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot' + extension)

        # markers
        markers = ['-', '--', '-.']
        # markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for k, key in enumerate(keys):
            line, = ax.plot(thr_iou,
                            performance[tracker_name][key]['succ_curve'],
                            markers[k % len(markers)])
            lines.append(line)
            legends.append('%s_%s: [%.3f]' % (
                'SOT3D Baseline', key, performance[tracker_name][key]['ao']))

        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

    def plot_curves(self, report_files, tracker_names, extension='.png'):
        assert isinstance(report_files, list), \
            'Expected "report_files" to be a list, ' \
            'but got %s instead' % type(report_files)

        # assume tracker_names[0] is your tracker
        report_dir = os.path.join(self.report_dir, tracker_names[0])
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

        performance = {}
        for report_file in report_files:
            with open(report_file) as f:
                performance.update(json.load(f))

        succ_file = os.path.join(report_dir, 'success_plot' + extension)
        key = 'overall'

        # filter performance by tracker_names
        performance = {k: v for k, v in performance.items() if k in tracker_names}

        # sort trackers by AO
        tracker_names = list(performance.keys())
        aos = [t[key]['ao'] for t in performance.values()]
        inds = np.argsort(aos)[::-1]
        tracker_names = [tracker_names[i] for i in inds]

        # markers
        markers = ['-', '--', '-.']
        markers = [c + m for m in markers for c in [''] * 10]

        # plot success curves
        thr_iou = np.linspace(0, 1, self.nbins_iou)
        fig, ax = plt.subplots()
        lines = []
        legends = []
        for i, name in enumerate(tracker_names):
            line, = ax.plot(thr_iou,
                            performance[name][key]['succ_curve'],
                            markers[i % len(markers)])
            lines.append(line)
            legends.append('%s: [%.3f]' % (
                name, performance[name][key]['ao']))
        matplotlib.rcParams.update({'font.size': 7.4})
        legend = ax.legend(lines, legends, loc='lower left',
                           bbox_to_anchor=(0., 0.))

        matplotlib.rcParams.update({'font.size': 9})
        ax.set(xlabel='Overlap threshold',
               ylabel='Success rate',
               xlim=(0, 1), ylim=(0, 1),
               title='Success plots')
        ax.grid(True)
        fig.tight_layout()

        # control ratio
        # ax.set_aspect('equal', 'box')

        print('Saving success plots to', succ_file)
        fig.savefig(succ_file,
                    bbox_extra_artists=(legend,),
                    bbox_inches='tight',
                    dpi=300)

parser = argparse.ArgumentParser("SOT 3D Baseline")
parser.add_argument('--dataset_root', default='E:\\SOT_3D', type=str, help='the root directory of SOT 3D dataset')
parser.add_argument('--model_path', default='models/model_e30.pth', type=str, help='the model path of SOT 3D baseline')
args = parser.parse_args()

if __name__ == '__main__':
    tracker = KCFTracker(True, True, True)
    experiment_sot3d = Experimet3DSOT(args.dataset_root, subset='test', use_dataset=True)
    experiment_sot3d.run(tracker, save_video=False, overwrite_result=True)
    experiment_sot3d.report([tracker.name])
    exit()
