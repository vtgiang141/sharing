import cv2
import time
import os, shutil
import imutils.video
from face_direction import DetectDirection

def get_dataset(name = "Giang"):
    # name = "Giang"
    # name = str(input('Name User: '))
    try:
        folder = "../Dataset/{}".format(name)
        # remove folder
        shutil.rmtree(folder, ignore_errors=False, onerror=None)
        # create tree-folder
    except:
        pass
    os.mkdir("../Dataset/{}".format(name))
    os.mkdir("../Dataset/{}/ID_CARD".format(name))
    os.mkdir("../Dataset/{}/model".format(name))
    os.mkdir("../Dataset/{}/processed".format(name))
    os.mkdir("../Dataset/{}/row".format(name))
    os.mkdir("../Dataset/{}/row/Unknow".format(name))
    os.mkdir("../Dataset/{a}/row/{a}".format(a=name))
    os.mkdir("../Dataset/{}/bounding_boxes".format(name))




    vs = imutils.video.VideoStream(src=2).start()
    images = []

    from distutils.dir_util import copy_tree

    # copy subdirectory example
    fromDirectory = "../Dataset/unknow"
    toDirectory = "../Dataset/{}/row/Unknow".format(name)

    ID_fromDirectory = "../Dataset/ID/{}.jpg".format(name)
    ID_toDirectory = "../Dataset/{}/ID_CARD".format(name)

    while True:

        frame = vs.read()
        time.sleep(0.3)
        # show the frame
        face_angle = DetectDirection(frame)

        pan = int(face_angle.get_pan_angle())
        rot = abs(int(face_angle.get_rot_angle()))
        
        # print("Pan angle: %s"%(pan), end="\t")
        # print("Rot angle: %s"%(rot))

        cv2.putText(frame, "Pan: "+str(pan), (150, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), lineType=cv2.LINE_AA)
        cv2.putText(frame, "Rot: "+str(rot), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), lineType=cv2.LINE_AA)


        if rot <= 10 and pan not in images and abs(pan) <= 15:
            images.append(pan)
            cv2.imwrite("../Dataset/{a}/row/{a}/{a}_{b}.jpg".format(a=name, b=pan), frame)

            cv2.putText(frame, "Image: "+str(len(images))+"/10", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), lineType=cv2.LINE_AA)
        cv2.putText(frame, "Image: "+str(len(images))+"/10", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), lineType=cv2.LINE_AA)
        
        cv2.imshow("Frame", frame)
        
        if len(images)==10:
            copy_tree(fromDirectory, toDirectory)
            # copy_tree(ID_fromDirectory, ID_toDirectory)
            shutil.copy(ID_fromDirectory, ID_toDirectory)

            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
        os.system('cls' if os.name == 'nt' else 'clear')

    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()