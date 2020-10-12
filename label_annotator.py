import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import argparse

attr_name = ["aspect-change", "dark-scene", "deformable", "depth-change",
             "fast-motion", "full-occlusion", "out-of-frame", "out-of-plane",
             "partial-occlusion", "reflective-target", "similar-objects",
             "size-change", "background-clutter", "unassigned"]

def read_rgbd(img_path, dp_path, frame_idx, out_data, depth_threshold=3000,
              txt_pos=(25, 25), font=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=1, fontColor=(255,255,255), lineType=2,
              box_color=(255,0,0), box_thickness=2):

    rgb = os.path.join(img_path, '%08d.jpg'%(frame_idx+1))                      # after rename
    depth = os.path.join(dp_path, '%08d.png'%(frame_idx+1))
    # rgb = os.path.join(img_path, '%06d_color.jpeg'%(frame_idx))                      # after rename
    # depth = os.path.join(dp_path, '%06d_depth.png'%(frame_idx))
    rgb = cv2.imread(rgb)
    cv2.putText(rgb,'frame idx : %d'%(frame_idx+1), txt_pos, font, fontScale,fontColor,lineType)
    depth = cv2.imread(depth, -1)                                               # cv2.IMREAD_GRAYSCALE)
    depth[depth>depth_threshold] = depth_threshold                              # ignore some large values, depth scale 1 = 1 mm
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth = cv2.applyColorMap(np.uint8(depth), cv2.COLORMAP_JET)
    I = cv2.hconcat((rgb, depth))
    I_temp = I.copy()

    rgb_box = out_data["groundtruth"][frame_idx, ...]
    dp_box = np.copy(rgb_box)
    dp_box[0] += 640
    cv2.rectangle(I, (int(rgb_box[0]), int(rgb_box[1])), (int(rgb_box[0]+rgb_box[2]), int(rgb_box[1]+rgb_box[3])), box_color, box_thickness)
    cv2.rectangle(I, (int(dp_box[0]), int(dp_box[1])), (int(dp_box[0]+dp_box[2]), int(dp_box[1]+dp_box[3])), box_color, box_thickness)

    return I, I_temp, rgb_box

def nothing(x):
    pass

def create_attr_panel():
    cv2.namedWindow("Attributes", cv2.WINDOW_NORMAL)
    for attr in attr_name:
        cv2.createTrackbar(attr, "Attributes", 0, 1, nothing)

def get_current_attr():
    current_values = {}
    for attr in attr_name:
        current_values[attr] = cv2.getTrackbarPos(attr, "Attributes")
    return current_values

def set_current_attr(attr_values, frame_idx):
    for attr in attr_name:
        cv2.setTrackbarPos(attr, "Attributes", attr_values[attr][frame_idx])

def update_out_data(out_data, box, frame_idx, out_path):
    attr_value = get_current_attr()
    for attr in attr_name:
        out_data[attr][frame_idx] = attr_value[attr]
    out_data["groundtruth"][frame_idx, ...] = box
    save_attributes_gt(out_data, out_path)
    return out_data # , attr_value

def load_data(seq_path, num_img):
    '''Load from groundtruth.txt and *.tag files'''
    data = {}

    for attr in attr_name:
        if os.path.isfile(os.path.join(seq_path, attr+".tag")):
            print('Tag: %s , load the existing file ...'%attr)
            with open(os.path.join(seq_path, attr+".tag"), 'r') as fp:
                lines = fp.readlines()
            data[attr] = np.array([int(v) for v in lines])
        else:
            print('\033[93m' + 'Tag: %s not found existing file !!!!!!!!!! '%attr + '\033[0m')
            data[attr] = np.zeros((num_img,), dtype=int)

    if os.path.isfile(os.path.join(seq_path, 'groundtruth.txt')):
        print('Init Groundtruth, load the existing file ...')
        with open(os.path.join(seq_path, 'groundtruth.txt'), 'r') as fp:
            lines = fp.readlines()
        gt_box = []
        for ii in range(len(lines)):
            line = lines[ii]
            gt_box.append([float(v) for v in line.split(",")])
        gt_box = np.array(gt_box)
        if len(gt_box) < num_img:
            padding = -1*np.ones((num_img-len(gt_box), 4), dtype=float)
            gt_box = np.concatenate((gt_box, padding), axis=0)
        data["groundtruth"] = gt_box

    else:
        print('\033[93m'+'Init Groundtruth :  not found existing file !!!!!!!!!!!!!' + '\033[0m')
        data["groundtruth"] = -1*np.ones((num_img, 4), dtype=float)
    print('----------------------------------------------')
    return data

def save_attributes_gt(data, output_path):
    '''Save groundtruth box and attriutes into .txt or .tag'''
    for key in attr_name:
        attr = data[key]
        with open(os.path.join(output_path, key+".tag"), 'w') as fp:
            for ii in range(len(attr) - 1):
                fp.write('%d\n'%attr[ii])
            fp.write('%d'%attr[-1])

        groundtruth = data["groundtruth"]
        with open(os.path.join(output_path, "groundtruth.txt"), 'w') as fp:
            for ii in range(len(groundtruth) - 1):
                fp.write('%f,%f,%f,%f\n'%(groundtruth[ii, 0], groundtruth[ii, 1], groundtruth[ii, 2], groundtruth[ii, 3]))
            fp.write('%f,%f,%f,%f'%(groundtruth[-1, 0], groundtruth[-1, 1], groundtruth[-1, 2], groundtruth[-1, 3]))

def attribute_annotator(sequence_path, sequences=None, out_path=None, depth_threshold=3000, box_color=(255,0,0), box_thickness=2):

    if not sequences:
        sequences = os.listdir(sequence_path)
    try:
        sequences.remove('list.txt')
    except:
        print('no list.txt ...')
    sequences.sort()

    for seq in sequences:
        img_path = os.path.join(sequence_path, seq, 'color')
        dp_path = os.path.join(sequence_path, seq, 'depth')
        if not out_path:
            out_path = os.path.join(sequence_path, seq)
        else:
            out_path = os.path.join(out_path, seq)
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        num_img = len(os.listdir(img_path))
        print('totally %d images'%num_img)

        '''Including, gt_box and attributes '''
        out_data = load_data(os.path.join(sequence_path, seq), num_img)

        '''Define the Mouse callback'''
        global ix, iy, drawing, box
        ix, iy, drawing = -1, -1, False

        '''Show RGB and Depth images'''
        frame_idx = 0
        I, I_temp, box = read_rgbd(img_path, dp_path, frame_idx, out_data, depth_threshold=depth_threshold, box_color=box_color, box_thickness=box_thickness)
        cv2.namedWindow(seq)
        cv2.imshow(seq, I)

        '''Create the panel for Attributes'''
        create_attr_panel()
        set_current_attr(out_data, frame_idx)

        def draw_box(event, x, y, flags, param):
            global ix, iy, drawing, box

            pos_shift = 640                                          # Image width

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing, ix, iy = True, x, y                         # init point, should be the Left Top Point
                box[0] = ix - pos_shift if ix > pos_shift else ix   # initilize the left top point
                box[1] = iy
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing == True:
                    if ix > pos_shift:
                        '''Mouse pointer on the Depth'''
                        rgb_lp_x = ix - pos_shift
                        dp_lp_x = ix
                        rgb_rp_x = x - pos_shift
                        dp_rp_x = x
                    else:
                        '''Mouse pointer on the RGB '''
                        rgb_lp_x = ix
                        dp_lp_x = ix + pos_shift
                        rgb_rp_x = x
                        dp_rp_x = x + pos_shift

                    box[2] = rgb_rp_x - rgb_lp_x
                    box[3] = y - iy
                    # print('%f, %f, %f, %f'%(rgb_lp_x, iy, rgb_rp_x, y))
                    I = I_temp.copy()
                    cv2.rectangle(I, (int(rgb_lp_x), int(iy)), (int(rgb_rp_x), int(y)), box_color, box_thickness)
                    cv2.rectangle(I, (int(dp_lp_x), int(iy)), (int(dp_rp_x), int(y)), box_color, box_thickness)
                    cv2.imshow(seq, I)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                if ix > pos_shift:
                    '''Mouse pointer on the Depth'''
                    rgb_lp_x = ix - pos_shift
                    dp_lp_x = ix
                    rgb_rp_x = x - pos_shift
                    dp_rp_x = x
                else:
                    '''Mouse pointer on the RGB '''
                    rgb_lp_x = ix
                    dp_lp_x = ix + pos_shift
                    rgb_rp_x = x
                    dp_rp_x = x + pos_shift

                box[2] = rgb_rp_x - rgb_lp_x
                box[3] = y - iy
                print('Frame - %d :  Box - '%(frame_idx+1), box)
                I = I_temp.copy()
                cv2.rectangle(I, (int(rgb_lp_x), int(iy)), (int(rgb_rp_x), int(y)), box_color, box_thickness)
                cv2.rectangle(I, (int(dp_lp_x), int(iy)), (int(dp_rp_x), int(y)), box_color, box_thickness)
                cv2.imshow(seq, I)

        cv2.setMouseCallback(seq, draw_box)

        while True:

            pressedKey = cv2.waitKey(1)

            if pressedKey == ord('s'):
                # save data for current frame
                out_data = update_out_data(out_data, box, frame_idx, out_path)

            if pressedKey == ord('q'):
                out_data = update_out_data(out_data, box, frame_idx, out_path)
                break

            if pressedKey == ord('a'):
                # save data for current frame
                out_data = update_out_data(out_data, box, frame_idx, out_path)

                # Go to the previous frame
                frame_idx = num_img-1 if frame_idx == 0 else frame_idx-1

                I, I_temp, box = read_rgbd(img_path, dp_path, frame_idx, out_data, depth_threshold=depth_threshold)
                cv2.imshow(seq, I)
                set_current_attr(out_data, frame_idx)

            elif pressedKey == ord('d'):
                # save data for current frame
                out_data = update_out_data(out_data, box, frame_idx, out_path)

                # Go to the next frame
                frame_idx = 0 if frame_idx == num_img-1 else frame_idx+1

                I, I_temp, box = read_rgbd(img_path, dp_path, frame_idx, out_data, depth_threshold=depth_threshold)
                cv2.imshow(seq, I)
                set_current_attr(out_data, frame_idx)

        cv2.destroyAllWindows()

parser = argparse.ArgumentParser(description='Settings for Annotator')
parser.add_argument('--sequence_path', type=str, default='/home/yan/Desktop/raw_results_crop/')
parser.add_argument('--sequences', nargs='+', default=None)
parser.add_argument('--out_path', type=str, default=None)
parser.add_argument('--depth_threshold', type=int, default=3000)

if __name__ == '__main__':

    args = parser.parse_args()
    # sequence_path =
    # sequence_path = '/home/yan/Data3/new_rgbd_dataset/realsense/raw_results_jinyu_1280x720/'
    # sequences = ['flower02_wild']
    print('Totally %d sequences'%len(args.sequences))

    attribute_annotator(args.sequence_path, sequences=args.sequences, out_path=args.out_path, depth_threshold=args.depth_threshold)
