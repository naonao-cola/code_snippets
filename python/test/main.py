import cv2
import os
import sys

###BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


BASE_DIR = R"D:\code\document"
sys.path.append(BASE_DIR)

# D:\data\element
video_root = os.path.join(BASE_DIR, '关键号码识别样本0705')

# open the folder
video_dirs = os.listdir(video_root)


def makedir(name, phase):
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


# Create a new folder to hold the new dataset
New_Folder = makedir(R"D:\code\py_ai\test\img", 'FireVideo')


def video_to_photo():
    # got all video files
    for index, video_file in enumerate(video_dirs):

        # got the video file root
        video_file = os.path.join(video_root, video_file)

        # read video
        cap = cv2.VideoCapture(video_file)

        # The number of videos cropped from the video
        count = 0
        Folder = makedir(New_Folder, str(index + 1))
        # frames
        fps = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # whether it is opened normally
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            if ret == True:
                count += 10
                if count < fps:
                    save_root = os.path.join(Folder, str(count) + '.jpg')
                    print("writing " + save_root + " ...")
                    cv2.imwrite(save_root, frame)
            cv2.waitKey(1)
    cap.release()


if __name__ == '__main__':
    video_to_photo()
    print("well done! ")
