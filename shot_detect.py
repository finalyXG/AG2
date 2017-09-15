from __future__ import division

import PySceneDetect.scenedetect
import ipdb as pdb
import numpy as np
import cv2
import scipy
from scipy import misc

def extract_shots_with_pyscenedetect(src_video, threshold=0, min_scene_length=15,  fps=25):
    """
    uses pyscenedetect to produce a list of shot 
    boundaries (in seconds)
    
    Args:
        src_video (string): the path to the source 
            video
        threshold (int): the minimum value used 
            by pyscenedetect to classify a shot boundary
        min_scene_length (int): the minimum number of frames
            permitted per shot. 
        fps (int): the frame rate of the video
    
    Returns: 
        List[(float, float)]: a list of tuples of floats 
        representing predicted shot boundaries (in seconds) and 
        their associated scores
    """
    scene_detectors = scenedetect.detectors.get_available()
    timecode_formats = scenedetect.timecodes.get_available()
    detection_method = 'content'
    detector = None
    start_time, duration, end_time = None, None, None
    
    # Setup scenedetect defaults
    downscale_factor = 1
    frame_skip = 0
    stats_writer = None
    quiet_mode, save_images = False, False
    
    detector = scene_detectors['content'](threshold, min_scene_length)
    scene_list = list()
    timecode_list = [start_time, duration, end_time]

    video_fps, frames_read = scenedetect.detect_scenes_file(
                            path = src_video,
                            #scene_list = scene_list,
                            detector_list = [detector],
                            stats_writer = stats_writer,
                            downscale_factor = downscale_factor,
                            frame_skip = frame_skip,
                            quiet_mode = quiet_mode,
                            save_images = save_images,
                            timecode_list = timecode_list)
    boundaries = [(pair[0] / fps, pair[1]) for pair in scene_list]
    return boundaries


class PySceneDetectArgs(object):
    def __init__(self, input, type='content'):
        self.input = input
        self.detection_method = type
        self.threshold = None
        self.min_percent = 95
        self.min_scene_len = 15
        self.block_size = 8
        self.fade_bias = 0
        self.downscale_factor = 1
        self.frame_skip = 2
        self.save_images = False
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.quiet_mode = True
        self.stats_file = None



def process():
    scene_detectors = scenedetect.detectors.get_available()
    path = "489.mp4"
    smgr_content   = scenedetect.manager.SceneManager(PySceneDetectArgs(input=path, type='content'),   scene_detectors)
    #smgr_threshold = scenedetect.manager.SceneManager(PySceneDetectArgs(input=video_path, type='threshold'), scene_detectors)

    scenedetect.detect_scenes_file(path=path, scene_manager=smgr_content)
    #scenedetect.detect_scenes_file(path=video_path, scene_manager=smgr_threshold)
    pdb.set_trace()

    detected_scenes = smgr_content.scene_list
    tmp = np.array(detected_scenes)
    pdb.set_trace()
    np.save('PySceneDetect/489_mp4_shot_boundary.h5.npy')

    pdb.set_trace()

def test():
    print 1


if __name__ == '__main__':
    process()
    print 1/0
    shot_len = 15
    bd = np.load('PySceneDetect/489_mp4_shot_boundary.h5.npy')
    count_end = bd[-1] - shot_len

    cap = cv2.VideoCapture('PySceneDetect/489.mp4')
    count = 0
    pdb.set_trace()
    for i in range(count_end):

        shot = []
        for j in range(i,i+15):
            if j in bd:
                shot = []
                break
            ret,frame = cap.read()
            frame = frame[0:650,0:1200]
            frame = cv2.resize(frame, (128, 128)) 
            shot.append(frame)



        pdb.set_trace()
        break
    
    # while cap.isOpened():
    #     ret,frame = cap.read()
        
        

        
    #     count = count + 1


    #     if cv2.waitKey(10) & 0xFF == ord('q'):
    #         pdb.set_trace()
    #         breakshosdkfj
    cap.release()
    # cap.destroyAllWindows()
