Lenet_Light
=============

### A highly customizable and convenient CNN structure based on python and tensorflow  

***
## Content  
* [Getting Started](#getting_started)
  * Prerequisites
  * Installing  
* [Running the model](#running_the_model)  
  * [Dataset](#dataset)
  * [Training](#training)
    * Original Training
    * Customize Training
  * [Testing](#testing)
    * Original Testing
    * Customize Testing

***
## Getting_Started
### Prerequisites
* python 3.6.5
* conda 4.5.4
* tensorflow 1.10.0
* opencv-python 3.4.2  
* Windows 7+ / Mac OS / Ubuntu 14.04+

### Installing
If you haven't installed anaconda on your computer, here is the URL link to [Download Anaconda](https://www.anaconda.com/download)
After installed `anaconda`, `tensorflow` and `opencv` can be installed by:
```
$ pip install tensorflow
$ pip install opencv-python
```
Make sure they are updated to the latest version. You can check their version by:
```
$ python -c 'import tensorflow as tf; print(tf.__version__)'
$ python -c 'import cv2; print(cv2.__version__)'
```

***
## Running the model
### Dataset
* Creating your own dataset is really convenient. In `data` directory there are two directories `train` and `test`.  

  All you have to do is to categorize your picture set in different directories respectively and store them in `train`.  

  `Lenet.py` will use the `name` of these directories as `labels` and train them accordingly. It is all the same for `test` dataset.  

  `cat` and `dog` directories are just examples, you can delete them after cloning this repository.
* Pictures **must** be in `.jpg` format.
* If you are using dataset collected by yourself, the dataset are preferably renamed and resized already.

### Training
#### Original Training
If you only want to use the simple Lenet model with 2 convolutional layers, all you have to do is to run `LeNet.py` by python:
```
$ python LeNet.py
```
Then the program will load all the images in `train` and `test` first. The reading process will be printed on the terminal:
```
reading the image:data/train/dog/dog.731.jpg
reading the image:data/train/dog/dog.514.jpg
reading the image:data/train/cat/cat.793.jpg
reading the image:data/train/cat/cat.996.jpg
reading the image:data/train/cat/cat.436.jpg
```
The training process will start as soon as all the images are successfully loaded.  
Depending on different configurations, the result of every epoch is going to show within different duration:
```
train loss: 1.8417279117135887
train acc:0.65571
test loss: 1.898901087524247
test acc:0.588808
```
The default epoch number is **50** times. After the training process, a `.pb` file will be generated in the original directory and know you can use `test.py` to access the `.pb` file.

If you want to use `tensorboard`, the `tfevent` file will be stored in `log` directory

#### Customize Training
I marked all the places that can be customized by `#TODO`  
* `#TODO 1`:
```python
  w = 32
  h = 32
  c = 3
```
Those are parameters for reshaping images. Depending on your computer configurations and your expected accuracy, you can change `w` and `h`. For colored pictures, `c` should be **3**(`RGB`)

* `#TODO 2`:
```python
pb_file_path = "Lenet.pb"
```
You can change the path for storing `.pb` file and the name of the file.

* **`#TODO 3`**:
All the major changes such as the number of `convolutional layers`, the number of `kernels` in each layers and the `size` of each `kernel` can all be modified here.  
The basic structure for one convolutional layers + one maxpooling layers is as followed:
```python
with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight',[5,5,c,6],initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias',[6],initializer=tf.constant_initializer(0.0))
        conv1 = tf.nn.conv2d(input_tensor,conv1_weights,strides=[1,1,1,1],padding='VALID')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))
with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
```
You can increase the number of layers by simply copy and paste the whole structure and change the name of those variables.  
Other parameters can also be changed according to your need.

* `TODO 4`:
`Full connected layers` can also be modified.  
The basic structure for one full connected layer is as followed:
```python
with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight',[nodes,120],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses',regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias',[120],initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped,fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1,0.5)
```

* `TODO 5` and `TODO 6`:
```python
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=y_)

cross_entropy_mean = tf.reduce_mean(cross_entropy)

loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
```
I am using `cross entropy mean` as my loss function and `AdamOptimizer` as my optimizer. You can change them into whatever you like.
****

* `TODO 7`:
```python
    train_num = 50
    batch_size = 12
```
Again, those two variables are decided by your computer configurations.

***
### Testing
#### Original Testing
Testing is much simpler.  
First, go to `test.py` and find `#TODO a`
```python
labels = {'[0]': 'cat', '[1]': 'dog'}
```
Change `keys` and `values` in `labels` according to the labels of your dataset  
Then find `#TODO d`
```python
if __name__ = '__main__':
    #recognize("img_path", "sample.pb")
```
Uncomment the `recognize` function and write the path of your testing img in `img path` and rename the `.pb` file.  
Then run `test.py` in your terminal:
```
$ python test.py
```
The predicted label should be shown on the terminal:
```
label: cat
```

#### Customize Testing
* `TODO c`:
You can add further application based on the predicted label or `import test` as a module in other python program.

****

|Author|SkyL|
|---|---
|E-mail|2924312854@qq.com

****
