# Traffic Sign Recognition

### Build a Traffic Sign Recognition Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_distribution.png "Class distribution"
[image2]: ./examples/all_signs.png "Sample images"
[image3]: ./examples/signs.png "Preprocessing - original"
[image4]: ./examples/signs_gray.png "Preprocessing - grayscale"
[image5]: ./examples/signs_equahist.png "Preprocessing - histogram equalization"
[image6]: ./examples/signs_localequahist.png "Preprocessing - local histogram equalization"
[image7]: ./new_test_images/s1.jpg "New sample 1"
[image8]: ./new_test_images/s2.jpg "New sample 2"
[image9]: ./new_test_images/s3.jpg "New sample 3"
[image10]: ./new_test_images/s4.jpg "New sample 4"
[image11]: ./new_test_images/s5.jpg "New sample 5"
[image12]: ./examples/feat_c1.png "Feat visual conv1"
[image13]: ./examples/feat_c2.png "Feat visual conv2"

### Project code

The project code is available on [GitHub](https://github.com/ggbalaazs/CarND-TrafficSigns-P2/blob/master/Traffic_Sign_Classifier.ipynb).

### Data Set Summary & Exploration

#### Basic summary 

The resized and pickled dataset can be downloaded from [here](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).  I used the numpy library to calculate summary statistics of the traffic signs data set:
```python
def summarize():
    global n_train, n_validation, n_test, image_shape, n_classes
    
    # number of training examples
    n_train = X_train.shape[0]
    # number of validation examples
    n_validation = X_valid.shape[0]
    # number of testing examples.
    n_test = X_test.shape[0]
    # shape of a traffic sign image
    image_shape = X_train.shape[1:]
    # unique classes/labels in the dataset.
    n_classes = np.unique(np.concatenate([train['labels'], valid['labels'], test['labels']])).shape[0]
```
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### Exploratory visualization

Here is a bar chart showing the class distribution over the whole set.  As we see this is far from a uniform distribution.

![alt text][image1]

This is a plot showing random sign of all classes.

![alt text][image3]

### Design and Test a Model Architecture

#### Preprocessing
 
I used the preprocessing steps as shown below:

```python
def preprocess(dataset):
    datares = np.ndarray(shape=(dataset.shape[0],32,32,1), dtype=np.float32)
    for i in range(0,dataset.shape[0]):
        img = dataset[i,]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = cv2.equalizeHist(img)
        #img = exposure.equalize_adapthist(img)*255
        img = ((img - 0.) / 255.).astype(np.float32)
        img = img.reshape(32, 32, 1)
        datares[i,] = img
    return datares
```

I decided to drop color information and experiment with grayscale images only. I used _OpenCV_ histogram equalization and also tried _skimage.exposure_ local adaptive histogram equalization. Though `exposure.equalize_adapthist` gave better visual results, it was computationally expensive, so I chose `equalizeHist`. Finally the images were normalized to `[0, 1]`.

Grayscale images
![alt text][image4]

Histogram equalization
![alt text][image5]

Local histogram equalization
![alt text][image6]


The preprocessing performed well, but for possibly better results YUV color coding could be explored as suggested in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Then again color channels would also need similar normalization.



#### Model architecture

My final model consisted of the following layers:

| Layer          | Description                                | 
| :-------------:| :-----------------------------------------:| 
| Input          | 32x32x1 grayscale image                    | 
| Convolution 5x5| 1x1 stride, valid padding, outputs 28x28x32|
| RELU           |                                            |
| Max pooling    | 2x2 stride,  outputs 14x14x32              |
| Convolution 5x5| 1x1 stride, valid padding, outputs 10x10x48|
| RELU           |                                            |
| Max pooling    | 2x2 stride,  outputs 5x5x48                |
| Flatten        | outputs 1200                               |
| Fully connected| outputs 400                                |
| RELU           |                                            |
| Dropout        | keep probability 0.75                      |
| Fully connected| outputs 84                                 |
| RELU           |                                            |
| Dropout        | keep probability 0.5                       |
| Fully connected| outputs 43                                 |
| Softmax        | etc.                                       |
 


#### Model training
In each epoch the training set was shuffled. `AdamOptimizer` was used to minimize the loss as proposed in the lectures.
```python
    'batch_size': 128,
    'max_epochs': 20,
    'learning_rate': 0.0005,
    # convolutional layers
    #  kernel     depth
    'c1_k': 5, 'c1_d': 32,
    'c2_k': 5, 'c2_d': 48,
    # fully connected layer
    #  depth       keep_prob
    'f1_d': 400, 'f1_kp': 0.75,
    'f2_d': 84,  'f2_kp': 0.5,
    'sigma': 0.1
```

#### Solution approach

Accuracy on training, validation and test sets were calculated as follows:
```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver2 = tf.train.import_meta_graph('./model_final.meta')
    saver2.restore(sess, "./model_final")
    test_accuracy = evaluate(X_test, y_test)
    print("Test Set Accuracy = {:.3f}".format(test_accuracy))
```
My final model results were:
* training set accuracy of 1.0
* validation set accuracy of 0.976
* test set accuracy of 0.95

Test set accuracy is close to validation set accuracy, so I think the model is working well.

Reaching this final model has been an iterative approach. At first I tried original LeNet model from previous lectures, but accuracy was around 87-91%. Changing epochs, batch size or learning rate did not really help. I was just experiencing different learning speeds with similar results. It was necessary to use a bigger model so I increased the depth of convolution and fully connected layers. Also adding dropout did really help. Although by mistake the dropout was also performed at evaluation in the first run. After eliminating this error accuracy went up to 95-97%. This seemed promising, so I moved on to evaluation on test set. It turned out to be 0.02%. Huge overfitting. 

To avoid overfitting I pruned the model and trained for less epochs. I reduced the second convolutional layer from `64` to `48` filters, the first fully connected layer output from `400` to `240` and decreased dropout keep probabilities from `0.8` `0.6` to `0.75` `0.5`. Max epochs was also decreased from 200 as validation accuracy had no benefit from it, but it led to overfitting eventually. Previously batch size was 256, memory was not an issue and it seemed fine. But with fewer the epochs using smaller batches seemed like a good idea. 

I also played with learning rate when accuracy improvement over epochs was not roughly monotonous but too much alternating.
 

### Test a Model on New Images

Here are five German traffic signs that I found on the web. They are already scaled to 32x32:

![alt text][image7]![alt text][image8]![alt text][image9]![alt text][image10]![alt text][image11]

These images have very tight crop and small to no surrounding. Aside from this difference visually they seem to be fine images with good contrast, except maybe the first. I expect no difficulties in classifying the images.

The prediction:
```python
new_labels = [12, 11, 1, 15, 28]
new_labels = np.asarray(new_labels, dtype=np.uint8)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver3 = tf.train.import_meta_graph('./model_final.meta')
    saver3.restore(sess, "./model_final")
    res = sess.run(tf.argmax(logits, 1), feed_dict={x: new_images, is_training: False})
``` 
Here are the results of the prediction:

| Image            | Prediction       | 
| :---------------:| :---------------:| 
| Priority road    | Priority road    | 
| Right-of-way     | Right-of-way     |
| 30 km/held       | 50 km/held       |
| No vehicles      | No vehicles      |
| Children crossing| Children crossing|

And the top five softmax probabilities:
```python
[[1.0000 0.0000 0.0000 0.0000 0.0000]
 [1.0000 0.0000 0.0000 0.0000 0.0000]
 [0.6469 0.1504 0.1191 0.0480 0.0149]
 [0.9999 0.0000 0.0000 0.0000 0.0000]
 [0.9983 0.0009 0.0006 0.0001 0.0000]]
 ```
 
 The classification failure expanded:
   * Speed limit (30km/h) is mistaken to Speed limit (50km/h)
   * Speed limit (30km/h) is mistaken to No passing for vehicles over 3.5 metric tons
   * Speed limit (30km/h) is mistaken to Speed limit (80km/h)
   * Speed limit (30km/h) is mistaken to Go straight or left
   * Speed limit (30km/h) is mistaken to Keep left

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. The misclassified image is 64.7% sure to be a similar traffic sign. The correctly classified cases are very certain, the model is not guessing here. Though 80% accuracy might not look good and it is much less than test set accuracy, evaluation on 5 images is not statistically significant at all. I think test accuracy is way more relevant here.

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is sure that this is priority road sign. 

| Probability         | Prediction    | 
| :-----------------: | :-----------: | 
| 1.0                 | Priority road |

For the second image, the model is sure that this is right-of-way sign. 

| Probability         | Prediction    | 
| :-----------------: | :-----------: | 
| 1.0                 | Right-of-way  |

For the third image, the top-5 softmax probabilities are the following:

| Probability| Prediction                                      | 
| :---------:| :---------------------------------------------: | 
| 0.647      | 50 km/held                                      | 
| 0.15       | No passing for vehicles over 3.5 metric tons    |
| 0.12       | 80 km/held                                      |
| 0.05       | Go straight or left                             |
| 0.01       | Keep left                                       |

For the fourth image, the model is quite sure that this is right-of-way sign. 

| Probability         | Prediction  | 
| :-----------------: | :---------: | 
| 0.9999              | No vehicles |


For the fifth image, the top softmax probabilities are the following:

| Probability| Prediction          | 
| :---------:| :------------------:| 
| 0.9983     | Children crossing   | 
| 0.0009     | Right-of-way        |
| 0.0006     | Dangerous left curve|
| 0.0001     | Pedestrians         |

### (Optional) Visualizing the Neural Network 

Weights of first convolutional layer visualized
![alt text][image12]

Weights of second convolutional layer visualized
![alt text][image13]

In second convolutional layer feature 26, 28, 33 and 42 seems like completely dead neurons with no response. Maybe the model could be further simplified without sacrificing accuracy.

