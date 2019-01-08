import numpy as np
import tensorflow as tf
import os
import cv2
import time
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",default='',type=str,help="The .pb model path")
parser.add_argument("--video_path",default='',type=str,help="The test video path,if None,use capture")


args = parser.parse_args()
PATH_TO_CKPT = args.model_path
video_path = args.video_path

if not os.path.exists(PATH_TO_CKPT):
    raise FileNotFoundError("The model file not found!")

PATH_TO_LABELS = {1: "open_eyes", 2: "close_eyes", 3: "phone", 4: "smoke", 5: "yawn", 6: "side_face", 7: "face"}

count = 0
NUM_CLASSES = 7
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
display=1

def load_graph(model_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


detection_graph = load_graph(PATH_TO_CKPT)
sess = tf.Session(graph=detection_graph)

if video_path:
    try:
        stream = cv2.VideoCapture(video_path)
    except:
        raise FileNotFoundError("The video can't open!")
elif not video_path:
    try:
        stream = cv2.VideoCapture(0)
    except:
        raise OSError("The capture is can't open")

while True:

    (grabbed,frame) = stream.read()

    if not grabbed:
        print("Finished")
        break
    start = time.time()
    pre_height = frame.shape[0]
    pre_width = frame.shape[1]
    img = cv2.resize(frame, (416, 416))

    # add N dim
    #input_data = np.expand_dims(img, axis=0)

    ops = detection_graph.get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    '''
    for key in [
        'detection_boxes', 'detection_scores',
        'detection_class', ]:'''
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
        ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = detection_graph.get_tensor_by_name(
                tensor_name)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(img, 0)})

    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    '''
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]'''

    for indx, boxes in enumerate(output_dict["detection_boxes"]):
        if output_dict["detection_scores"][indx]  > 0.8 and output_dict["detection_scores"][indx]  <=1.0:
            top, left, bottom, right = boxes
            top = int(top * pre_height)
            right = int(right * pre_width)
            bottom = int(bottom * pre_height)
            left = int(left * pre_width)
            try:
                cv2.rectangle(frame, (left, top), (right, bottom),
                              (0, 255, 0), 2)
            except:
                continue

            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, PATH_TO_LABELS[int(output_dict["detection_classes"][indx])] + ":" + str(output_dict['detection_scores'][indx]),
                            (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
            print("frame:{}".format(count),str(PATH_TO_LABELS[int(output_dict["detection_classes"][indx])])+":"+ \
                  str(output_dict['detection_scores'][indx] * 100)+"%")
    count+=1
    if display > 0:
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(50) & 0xFF
        end = time.time() - start
        #print("cost_{}s".format(end))
        #print(str(end)+"s")
        if key == ord("q"):
            break
sess.close()
