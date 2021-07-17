# -*- coding: utf-8 -*-


##Constructing the model##

import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers import ZeroPadding2D 
from tensorflow.python.keras.layers import Conv2D
from tensorflow.python.keras.layers import BatchNormalization 
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras import Model
import cv2

class BatchNormalization(tf.keras.layers.BatchNormalization):
	def call(self, x, training= False):
		if not training:
			training = tf.constant(False)
		training = tf.logical_and(training, self.trainable)
		return super().call(x, training)




def convolutional(input_layer, filters_shape, down_sample = False,
		activate = True, batch_norm = True, regularization = 0.0005, reg_stddev = 0.01, activate_alpha = 0.1):

	if down_sample:
		input_layer = tf.keras.layers.ZeroPadding2D(((1,0),(1,0)))(input_layer)
		padding ="valid"
		strides = 2
	else:
		padding ="same"
		strides = 1
	conv = tf.keras.layers.Conv2D(filters=filters_shape[-1],
		kernel_size = filters_shape[0],
		strides = strides,
		padding = padding,
		use_bias = not batch_norm,
		kernel_regularizer= tf.keras.regularizers.l2(regularization),
		kernel_initializer = tf.random_normal_initializer(stddev=reg_stddev),
		bias_initializer = tf.constant_initializer(0.)
		)(input_layer)

	if batch_norm:
		conv = BatchNormalization()(conv)
	if activate:
		conv = tf.nn.leaky_relu(conv, alpha= activate_alpha)

	return conv

def res_block(input_layer, input_channel, filter_num1, filter_num2):
	short_cut = input_layer
	conv = convolutional(input_layer, filters_shape=(1,1,input_layer,filter_num1))
	conv = convolutional(conv, filters_shape=(3,3,filter_num1,filter_num2))

	res_output = short_cut+ conv 
	return res_output

def darknet53(input_data):
	input_data = convolutional(input_data,(3,3,3,32))
	input_data = convolutional(input_data, (3,3,32,64), down_sample = True)

	for i in range(1):
		input_data = res_block(input_data, 64,32,64)

	input_data = convolutional(input_data, (3,3,64,128),down_sample=True)

	for i in range(2):
		input_data = res_block(input_data, 128,64,128)

	input_data = convolutional(input_data, (3,3,128,256), down_sample= True)

	for i in range(8):
		input_data = res_block(input_data,256,128,256)


	route_1 = input_data 

	input_data = convolutional(input_data,(3,3,256,512), down_sample= True)

	for i in range(8):
		input_data = res_block(input_data,512,256,512)
	route_2 = input_data
	input_data = convolutional(input_data,(3,3,512,1024), down_sample= True)

	for i in range(4):
		input_data= res_block(input_data,1024,512,1024)


	return route_1, route_2, input_data

def upsample(input_layer):
	return tf.image.resize(input_layer,(input_layer.shape[1]*2,input_layer.shape[2]*2),
		method='nearest')

# hyperparameters 
NUM_CLASSES = 80
STRIDES = np.array([8,16,32])
ANCHORS =(1.25,1.625, 2.0,3.75, 4.125,2.875, 1.875,3.8125, 3.875,2.8125, 3.6875,7.4375, 3.625,2.8125, 4.875,6.1875, 11.65625,10.1875)
ANCHORS = np.array(ANCHORS).reshape(3,3,2)
weight_file = "yolov4.weights"

def yoloV4(input_layer):
	route_1, route_2, conv = darknet53(input_layer) 

	conv = convolutional(conv, (1,1,1024,512)) 
	conv = convolutional(conv,(3,3,512,1024)) 
	conv = convolutional(conv, (1,1,1024, 512))  
	conv = convolutional(conv, (3,3,512,1024))  
	conv = convolutional(conv,(1,1,1024,512)) 

	conv_lobj_branch = convolutional(conv,(3,3,512,1024))
	conv_lbbox = convolutional(conv_lobj_branch,(1,1,1024,3*(NUM_CLASSES+5)),
		activate= False, batch_norm = False)

	conv = convolutional(conv,(1,1,512,256))
	conv = upsample(conv)

	conv = tf.concat([conv, route_2], axis =-1) 
	conv = convolutional(conv,(1,1,768,256)) 
	conv = convolutional(conv,(3,3,256, 512))
	conv = convolutional(conv,(1,1,512,256))
	conv = convolutional(conv,(3,3,256,512))
	conv = convolutional(conv, (1,1,512,256))

	conv_mobj_branch = convolutional(conv, (3,3,256,512))
	conv_mbbox = convolutional(conv_mobj_branch ,(1,1,512,3*(NUM_CLASSES+5)),
		activate= False, batch_norm= False)


	conv = convolutional(conv, (1,1,256,128))
	conv = upsample(conv)

	conv = tf.concat([conv,route_1], axis = -1)

	conv = convolutional(conv, (1,1,384,128))
	conv = convolutional(conv, (3,3,128, 256))
	conv = convolutional(conv, (1,1,256, 128))
	conv = convolutional(conv, (3,3,128, 256))
	conv = convolutional(conv, (1,1,256, 128))

	conv_sobj_branch = convolutional(conv,(3,3,128, 256))
	conv_sbbox = convolutional(conv_sobj_branch,
		(1,1,256,3*(NUM_CLASSES+5)),activate= False , batch_norm= False)
	return [conv_sbbox, conv_mbbox, conv_lbbox]


def decode(conv_out, i = 0):
	conv_shape = tf.shape(conv_out)
	batch_size = conv_shape[0]
	output_size = conv_shape[1]

	conv_output = tf.reshape(conv_out, (batch_size, output_size,output_size, 3,5+NUM_CLASSES))
	
	conv_raw_dxdy = conv_output[:,:,:,:,0:2]
	conv_raw_dwdh = conv_output[:,:,:,:,2:4]
	conv_raw_conf = conv_output[:,:,:,:,4:5]
	conv_raw_prob = conv_output[:,:,:,:,5:]

	y = tf.tile(tf.range(output_size,dtype=tf.int32)[:,tf.newaxis],[1,output_size])
	x = tf.tile(tf.range(output_size, dtype= tf.int32)[tf.newaxis,:],[output_size,1])

	xy_grid = tf.concat([x[:,:,tf.newaxis],y[:,:,tf.newaxis]], axis = -1)
	xy_grid = tf.tile(xy_grid[tf.newaxis,:,:,tf.newaxis,:],[batch_size,1,1,3,1])
	xy_grid = tf.cast(xy_grid,tf.float32)

	pred_xy = (tf.sigmoid(conv_raw_dxdy)+xy_grid)*STRIDES[i]
	pred_wh = (tf.exp(conv_raw_dwdh)*ANCHORS[i])*STRIDES[i]
	pred_xywh = tf.concat([pred_xy,pred_wh], axis = -1)

	pred_conf = tf.sigmoid(conv_raw_conf)
	pred_prob = tf.sigmoid(conv_raw_prob)

	return tf.concat([pred_xywh, pred_conf, pred_prob], axis = -1)
 
def Load_weights(model,weight_file):

    wf = open(weight_file, 'rb')
    major , minor, revision , seen, _ = np.fromfile(wf,dtype= np.int32, count=5)
    j=0

    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i>0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j>0 else 'batch_normalization' 

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]


        if i not in [58,66,74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype= np.float32, count = 4*filters)
            bn_weights = bn_weights.reshape((4,filters))[[1,0,2,3]]
            bn_layer = model.get_layer(bn_layer_name)

            j+=1

        else:
            conv_bias = np.fromfile(wf,dtype= np.float32, count= filters)

        # darknet shape is (out_dim, in_dim, height,width)
        conv_shape = (filters, in_dim,k_size,k_size)
        conv_weights = np.fromfile(wf,dtype= np.float32, count= np.product(conv_shape))

        #tf shpae (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2,3,1,0])


        if i not in [58,66,74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights,conv_bias])

    assert len(wf.read(0))==0, 'failed to read all data'
    wf.close()

    return model

def Model():
	input_layer = tf.keras.layers.Input([416,416,3])
	feature_maps = yoloV4(input_layer)

	bbox_tensors = []

	for i , fm in enumerate(feature_maps):
		bbox_tensor = decode(fm, i)
		bbox_tensors.append(bbox_tensor)


	model = tf.keras.Model(input_layer, bbox_tensors)
	model = Load_weights(model, weight_file)

	return model 

    
print('Done initializing functions')

##Preprocessing and Detection Functions##

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def transform_images(x_train, size):
    x_train = tf.image.resize(x_train, (size, size))
    x_train = x_train / 255
    return x_train

def box_detector(pred):
    center_x,center_y,width,height,confidence,classes = tf.split(pred,[1,1,1,1,1,-1], axis=-1)
    top_left_x=(center_x-width/2.)/ 416
    top_left_y = (center_y - height / 2.0)/416.0
    bottom_right_x = (center_x + width / 2.0)/416.0
    bottom_right_y = (center_y + height / 2.0)/416.0
    #pred = tf.concat([top_left_x, top_left_y, bottom_right_x,
    #bottom_right_y, confidence, classes], axis=-1)

    boxes = tf.concat([top_left_y,top_left_x,bottom_right_y,bottom_right_x],axis=-1)
    scores = confidence*classes
    scores = np.array(scores)

    scores = scores.max(axis=-1)
    class_index = np.argmax(classes, axis=-1)

    final_indexes = tf.image.non_max_suppression(boxes,scores, max_output_size= 20)
    final_indexes = np.array(final_indexes)
    class_names = class_index[final_indexes]
    boxes = np.array(boxes)
    scores = np.array(scores)
    class_names = np.array(class_names)
    boxes = boxes[final_indexes,:]

    scores = scores[final_indexes]
    boxes = boxes*416

    return boxes ,class_names, scores

def drawbox(boxes, class_names,scores,names,img):
    data = np.concatenate([boxes,scores[:,np.newaxis],class_names[:,np.newaxis]],axis=-1)
    data = data[np.logical_and(data[:, 0] >= 0, data[:, 0] <= 416)]
    data = data[np.logical_and(data[:, 1] >= 0, data[:, 1] <= 416)]
    data = data[np.logical_and(data[:, 2] >= 0, data[:, 2] <= 416)]
    data = data[np.logical_and(data[:, 3] >= 0, data[:,3] <= 416)]
    data = data[data[:,4]>0.4]
    

    img = cv2.resize(img, (416, 416))
    person = 0
    for i,row in enumerate(data):
        #print(row)
        #print(data)
        #print(names[row[5]])
       
        if names[row[5]]=="person" or names[row[5]]=="bottle" or names[row[5]]=="chair" or names[row[5]]=="book" or names[row[5]]=="cell phone" or names[row[5]]=="backpack":
            t_size = cv2.getTextSize(names[row[5]] ,cv2.FONT_HERSHEY_PLAIN, 0.48 , 1)[0]
            img = cv2.rectangle(img, (int(row[1]),int(row[0] - t_size[1]-3)),(int(row[1]+t_size[0] + 3),int(row[0])), 
                            (206,0,0),-1)
            img = cv2.rectangle(img,(int(row[1]),int(row[0])),(int(row[3]),int(row[2])) ,(206,209,0),1)
            img = cv2.putText(img, names[row[5]],(int(row[1]),int(row[0]-3)), cv2.FONT_HERSHEY_PLAIN,
                          0.48,(255,255,255 ),1)
        

    return  img


def main():

    model = Model()
    names= read_class_names("classes.names")
    # img = cv2.imread("download.jpeg")
    # img_in = tf.expand_dims(img,0)
    # img_in = transform_images(img_in, 416)
    # pred_bbox = model.predict(img_in)
    # pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    # pred_bbox = tf.concat(pred_bbox, axis=0)

    # boxes ,class_names, scores = box_detector(pred_bbox)
    # img = drawbox(boxes ,class_names, scores,names,img)

    # img = cv2.resize(img, (1200, 700))
    # cv2_imshow(img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    cap = cv2.VideoCapture("Scene_understanding.mp4")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID") 
    fps = int(cap.get(cv2.CAP_PROP_FPS))

# For saving videos
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width,height), True)
    while cap.isOpened():
        ret, img = cap.read()    
        if img is None:
            print("empty frames")
            continue

        img_in = tf.expand_dims(img,0)
        img_in = transform_images(img_in, 416)
        pred_bbox = model.predict(img_in)
        pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis=0)

        boxes ,class_names, scores = box_detector(pred_bbox)
        img= drawbox(boxes ,class_names, scores,names,img)
        
        out.write(img)
        img = cv2.resize(img, (1200,720))
        cv2.imshow("output", img)
        
        if cv2.waitKey(1) == ord("q"):
            break
    
    out.release
    cv2.destroyAllWindows()

	
main()

# from google.colab import drive
# drive.mount('/content/drive')

#!pip install face_recognition

