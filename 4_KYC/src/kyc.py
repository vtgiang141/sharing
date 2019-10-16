from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, shutil
import cv2
import sys
import tensorflow as tf
import numpy as np
import facenet
import align.detect_face
import random
from time import sleep
import math
import pickle
from sklearn.svm import SVC
import collections
import matplotlib.pyplot as plt
import timeit
import matplotlib.image as mpimg

from capture import get_dataset

# name = "Phap"
name = str(input("\nName: "))

get_dataset(name)
data_path = "../Dataset/{}/".format(name)

# for name in os.listdir("../Dataset/"):
#     data_path = "../Dataset/{}/".format(name)

#   print("__________ name = ", name)
# sys.exit(0)
    # delete files
def remove_processed():
    folder = '../Dataset/{}/processed/'.format(name)
    # shutil.rmtree(folder, ignore_errors=False, onerror=None)
    for file in os.listdir(folder):
        try:
            shutil.rmtree(folder+file, ignore_errors=False, onerror=None)
        except:
            pass
    try:
        os.remove(folder+"revision_info.txt")
    except:
        pass
    print("delete all in processed folder!")

# change file name 
def change_names():
    folder = '../Dataset/{}/row/'.format(name)
    print("len row folder: ", len(os.listdir(folder)))
    for file in os.listdir(folder):
        path = folder + file
        for num,image in enumerate(os.listdir(path)):
            num+=1
            src=path+'/'+image
            print(src)

            dst=path+'/'+str(num)+".jpg"
            print(dst)
            os.rename(src,dst)


# find bounding boxes
def bounding_boxes(data_path="../Dataset/{}/".format(name)):
    if not os.path.exists( data_path+'processed' ):
        os.makedirs( data_path+'processed' )
    # Store some git revision info in a text file in the log directory
    src_path = os.path.abspath('')
    facenet.store_revision_info(src_path, data_path+'processed', 'xx')
    print("number of classes in DB: ", len(facenet.get_dataset(data_path+'row')))

    print('Creating networks and loading parameters')

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.70)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)

    # Add a random key to the filename to allow alignment using multiple processes
    random_key = np.random.randint(0, high=99999)
    bounding_boxes_filename = os.path.join(data_path+"bounding_boxes", 'bounding_boxes_%05d.txt' % random_key)
    return [bounding_boxes_filename, pnet, rnet, onet]


# count all files in row folder
def count_files(data_path="../Dataset/{}/".format(name)):
    num_image = 0
    for each_fol in os.listdir(data_path+'row'):
        for each in os.listdir(data_path+'row/'+each_fol):
            num_image +=1
    return num_image

bounding_boxes_filename, pnet, rnet, onet = bounding_boxes(data_path)

# get image and save to 
def feature_extraction(minsize = 20, 
                    threshold = [ 0.6, 0.7, 0.7 ], 
                    factor = 0.709, 
                    bounding_boxes_filename=bounding_boxes_filename,
                    pnet = pnet,
                    rnet = rnet,
                    onet = onet):

    start = timeit.default_timer()

    dataset = facenet.get_dataset(data_path+'row')
    
    num_files = count_files()
    count = 0
    with open(bounding_boxes_filename, "w") as text_file:
        nrof_images_total = 0
        nrof_successfully_aligned = 0
        if '--random_order':
            random.shuffle(dataset)
        
        for cls in dataset:
            output_class_dir = os.path.join(data_path+'processed', cls.name)
            if not os.path.exists(output_class_dir):
                os.makedirs(output_class_dir)
                if '--random_order':
                    random.shuffle(cls.image_paths)

            for image_path in cls.image_paths:
                nrof_images_total += 1
                filename = os.path.splitext(os.path.split(image_path)[1])[0]
                output_filename = os.path.join(output_class_dir, filename+'.png')
                print(image_path)

                if not os.path.exists(output_filename):
                    try:
                        img = cv2.imread(image_path)
                    except (IOError, ValueError, IndexError) as e:
                        errorMessage = '{}: {}'.format(image_path, e)
                        print(errorMessage)
                    else:
                        if img.ndim<2:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))
                            continue
                        if img.ndim == 2:
                            img = facenet.to_rgb(img)
                        img = img[:,:,0:3]

                        bounding_boxes, _ = align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
                        nrof_faces = bounding_boxes.shape[0]

                        if nrof_faces>0:
                            det = bounding_boxes[:,0:4]
                            det_arr = []
                            img_size = np.asarray(img.shape)[0:2]

                            if nrof_faces>1:
                                if False:
                                    for i in range(nrof_faces):
                                        det_arr.append(np.squeeze(det[i]))
                                else:
                                    bounding_box_size = (det[:,2]-det[:,0])*(det[:,3]-det[:,1])
                                    img_center = img_size / 2
                                    offsets = np.vstack([ (det[:,0]+det[:,2])/2-img_center[1], (det[:,1]+det[:,3])/2-img_center[0] ])
                                    offset_dist_squared = np.sum(np.power(offsets,2.0),0)
                                    index = np.argmax(bounding_box_size-offset_dist_squared*2.0) # some extra weight on the centering
                                    det_arr.append(det[index,:])
                            else:
                                det_arr.append(np.squeeze(det))

                            margin = 32
                            image_size = 160

                            for i, det in enumerate(det_arr):
                                det = np.squeeze(det)
                                bb = np.zeros(4, dtype=np.int32)
                                bb[0] = np.maximum(det[0]-margin/2, 0)
                                bb[1] = np.maximum(det[1]-margin/2, 0)
                                bb[2] = np.minimum(det[2]+margin/2, img_size[1])
                                bb[3] = np.minimum(det[3]+margin/2, img_size[0])
                                cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
                                scaled = cv2.resize(cropped, (image_size, image_size), cv2.INTER_LINEAR)
                                nrof_successfully_aligned += 1
                                filename_base, file_extension = os.path.splitext(output_filename)
                                if False:
                                    output_filename_n = "{}_{}{}".format(filename_base, i, file_extension)
                                else:
                                    output_filename_n = "{}{}".format(filename_base, file_extension)
                                cv2.imwrite(output_filename_n, scaled)
                                count += 1

                                # clear_output()
                                os.system('cls' if os.name == 'nt' else 'clear')

                                print("{} - finish: ".format(name), str(float(count/num_files*100)).split(".")[0]+"."+str(count/num_files*100).split(".")[1][:2], " %")
                                
                                text_file.write('%s %d %d %d %d\n' % (output_filename_n, bb[0], bb[1], bb[2], bb[3]))
                        else:
                            print('Unable to align "%s"' % image_path)
                            text_file.write('%s\n' % (output_filename))

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)
    stop = timeit.default_timer()

    print('Time: ', stop - start)  

def training(data_path="../Dataset/{}/".format(name)):
    start = timeit.default_timer()
    
    with tf.Graph().as_default():

        with tf.Session() as sess:

            dataset = facenet.get_dataset(data_path+'processed')
            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model('../Models/facenet/20180402-114759.pb')

            print("-- get I/O of tensors")
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / 1000))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*1000
                end_index = min((i+1)*1000, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, 160)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = "../Dataset/{a}/model/{b}.pkl".format(a=name, b=name)

            print('Training classifier')
            model = SVC(kernel='linear', probability=True)
            model.fit(emb_array, labels)

            # Create a list of class names
            class_names = [ cls.name.replace('_', ' ') for cls in dataset]

            # Saving classifier model
            with open(classifier_filename_exp, 'wb') as outfile:
                pickle.dump((model, class_names), outfile)
            print('Saved classifier model to file "%s"' % classifier_filename_exp)
            stop = timeit.default_timer()
            time = str(float(stop - start)).split(".")
            print('Time: '+time[0]+"."+time[1][:2])


def recognition( MINSIZE = 20,
                    THRESHOLD = [0.6, 0.7, 0.7],
                    FACTOR = 0.709,
                    IMAGE_SIZE = 182,
                    INPUT_IMAGE_SIZE = 160):

    print("\n__Start recognition__\n")
    # Load The Custom Classifier
    with open("../Dataset/{a}/model/{b}.pkl".format(a=name, b=name), 'rb') as file:
        model, class_names = pickle.load(file)
        print("path: " + "../Dataset/{a}/model/{b}.pkl".format(a=name, b=name))
        print("model:    ", model)
        print("class_names:     ", class_names)
    print("Custom Classifier, Successfully loaded")


    with tf.Graph().as_default():
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        
        with sess.as_default():
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model('../Models/facenet/20180402-114759.pb')
            # facenet.load_model("20180402-114759.pb")
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "align")
            
            people_detected = set()
            person_detected = collections.Counter()
            
            image_path = "../Dataset/{a}/ID_CARD/{b}.jpg".format(a=name, b=name)
            
            print("imge path: ", image_path)
            frame = mpimg.imread(image_path)
            # frame = cv2.rotate(frame,rotateCode = 2)
            
            bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            faces_found = bounding_boxes.shape[0]

            det = bounding_boxes[:, 0:4]
            bb = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                bb[i][0] = det[i][0]
                bb[i][1] = det[i][1]
                bb[i][2] = det[i][2]
                bb[i][3] = det[i][3]

                cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                
                scaled = facenet.prewhiten(scaled)
                scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                emb_array = sess.run(embeddings, feed_dict=feed_dict)
                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                best_name = class_names[best_class_indices[0]]

                print("\n\nResult: \n==================================================")
                print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (255, 255, 0), 2)
                text_x = bb[i][0]
                text_y = bb[i][3] + 20
                print("--", text_x, "--", text_y)
                cv2.putText(frame, best_name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 255), thickness=1, lineType=2)
                cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), thickness=1, lineType=2)
                
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imwrite("../output/{}.jpg".format(name), frame)
            # cv2.imwrite(0)

pretrain = True

if pretrain:
    remove_processed()

if pretrain:
    feature_extraction()
    training()
recognition()
print("\n==================================================")
print("\n Done {} \n".format(name))