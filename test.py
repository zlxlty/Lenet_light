import tensorflow as tf
import  numpy as np
import PIL.Image as Image
from skimage import io, transform
import cv2

def recognize(jpg_path, pb_file_path):
    
    labels = {'[0]': 'blue', '[1]': 'green', '[2]': 'red'}
    
    w=64
    d=64
    c=3
    
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()

        with open(pb_file_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(output_graph_def, name="")

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            input_x = sess.graph.get_tensor_by_name("input:0")
            #print (input_x)
            out_softmax = sess.graph.get_tensor_by_name("softmax:0")
            #print (out_softmax)
            out_label = sess.graph.get_tensor_by_name("output:0")
            #print (out_label)

            img = io.imread(jpg_path)
#            cv2.namedWindow('showimage')
#            cv2.imshow("Image", img)
#            cv2.waitKey(0)
#            cv2.destroyAllWindows()
            #cv2.namedWindow('showimage')
        
            imged = transform.resize(img, (w, d, c))
            img_out_softmax = sess.run(out_softmax, feed_dict={input_x:np.reshape(imged, [-1, w, d, c])})

            #print ("img_out_softmax:",img_out_softmax)
            prediction_labels = np.argmax(img_out_softmax, axis=1)
            label = labels[str(prediction_labels)]
        
            #font = cv2.FONT_HERSHEY_SIMPLEX
            #cv2.putText(img, label, (50,50), font, 0.8, (0,255,0), 2)
            #img = io.imread(jpg_path)
            
            #cv2.imshow("Image", img)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            print ("label:",label)
            #print('\n')


recognize("unknown4.jpg", "sky.pb")
recognize("5.jpg", "sky.pb")
recognize("6.jpg", "sky.pb")
recognize("7.jpg", "sky.pb")


