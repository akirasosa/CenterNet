import os

import cv2

# noinspection PyUnresolvedReferences
import _init_paths
from detectors.detector_factory import detector_factory
from opts import opts

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)

    if opt.demo == 'webcam' or \
            opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
        cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)
        detector.pause = False
        while True:
            _, img = cam.read()
            cv2.imshow('input', img)
            ret = detector.run(img)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)
            if cv2.waitKey(1) == 27:
                return  # esc to quit
    else:
        if os.path.isdir(opt.demo):
            image_names = []
            ls = os.listdir(opt.demo)
            for file_name in sorted(ls):
                ext = file_name[file_name.rfind('.') + 1:].lower()
                if ext in image_ext:
                    image_names.append(os.path.join(opt.demo, file_name))
        else:
            image_names = [opt.demo]

        for (image_name) in image_names:
            ret = detector.run(image_name)
            time_str = ''
            for stat in time_stats:
                time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
            print(time_str)


if __name__ == '__main__':
    opt = opts().init()
    # import pickle
    # with open('../data/opt.pkl', 'wb') as f:
    #     pickle.dump(opt, f)
    # with open('/home/akirasosa/.ghq/github.com/akirasosa/CenterNet/data/coco/opt.pkl', 'rb') as f:
    #     opt = pickle.load(f)
    # opt.arch = 'mobile'
    # opt.load_model = '/home/akirasosa/.ghq/github.com/akirasosa/CenterNet/exp/multi_pose/mobile/model_best.pth'
    # opt.load_model = '/home/akirasosa/.ghq/github.com/akirasosa/CenterNet/exp/multi_pose/res18/model_best.pth'
    demo(opt)
