import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import io, transform
import cv2

def recognize(jpg_path, pb_file_path):

    #TODO a   keys and values in labels can be customized according to your dataset
    labels = {'[0]': 'blue', '[1]': 'green', '[2]': 'red'}

    #TODO b  adject the height, width and channel according to your picture (better be the same as ones in Lenet.py)
    w=64
    d=64
    c=3

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        # read your pre-trained model from .pb file
        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # load all the parameters
            input_x = sess.graph.get_tensor_by_name("input:0")
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            out_label = sess.graph.get_tensor_by_name("output:0")

            # read img from jpg_path
            img = io.imread(jpg_path)

            # reshape the img according to the setting above
            imged = transform.resize(img, (w, d, c))
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(imged, [-1, w, d, c])})

            # get prediction
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            label = labels[str(prediction_labels)]

            #TODO c  you can use this label to realize other functions
            print ("label:",label)

#TODO d  recognize function can be used here or be imported by other modules
if __name__ = '__main__':
    #recognize("img_path", "sample.pb")
