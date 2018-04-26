import numpy as np
import os, sys
#from CNN import eval
from cntk.ops.functions import load_model
from PIL import Image 
import scipy.misc
import cv2
import time




from IPython.display import Image as display 

label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

z = load_model('../../../DNN_Project_Implementation\pred_vgg9_300_400.dnn')

#print(z.architecture())

#img_file = resizeIt('Car.png')
#Car_3.png
rgb_image = np.asarray(Image.open("_Car_small.jpg"), dtype=np.float32) - 128
bgr_image = rgb_image[..., [2, 1, 0]]
pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

predictions = np.squeeze(z.eval({z.arguments[0]:[pic]}))
top_class = np.argmax(predictions)

#print("\tTOP CLASS Label: {:10s}, confidence: {:.2f}%".format(label_lookup[top_class], predictions[top_class] * 100))

#FPS:134.04600493493618 can be processed
# Display Image
#Image.open("Car.png").show("Car")
#display("Car.png", width=300, height=300,embed=True)

def evaluate_model(pred_op, image_path):
	label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	st_time = time.clock()
	bgr_image = np.asarray(Image.open(image_path), dtype=np.float32) - 127.5
	bgr_image = bgr_image[..., [2, 1, 0]]
	pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
	
	result = np.squeeze(pred_op.eval({pred_op.arguments[0]:[pic]}))
	
	name =""
	names = ['Automobile','Airplane','Dog','Horse','Car','Ship','Truck','Bird']
	test_name = '_PDHCSTB'
	name = names[test_name.find(image_path[0])]
    # Return top 3 results:
	top_count = 3
	result_indices = (-np.array(result)).argsort()[:top_count]
	end_time = time.clock()
	print("Top 3 predictions:")
	vowels ='oiuea'
	for i in range(top_count):
		label = label_lookup[result_indices[i]]
		conf = result[result_indices[i]] * 100
		article = 'an'
		if vowels.find(label_lookup[result_indices[i]][:1]) == -1:
			article ='a'
		if  result[result_indices[i]] * 100 >= 99.0:
			print("\tConfident {:s} {:8s}, confidence: {:.2f}%".format(article,label, conf))
		elif result[result_indices[i]] * 100 > 88.0:
			print("\tMost likely {:s} {:7s}, confidence: {:.2f}%".format(article,label, conf))
		else:
			print("\tLabel: {:14s}, confidence: {:.2f}%".format(label, conf))
			
	print('Compute time was {:.10f}, and OBJECT was {:s} with name:{:s}'.format(end_time-st_time,name,image_path))
	return (end_time-st_time)

files = ['_Car_small.jpg','P_sm.png','P_2.png','T_1_s.jpg','H_1_s.jpg','D_3_Boy_s.png'
	#,'D_d_small.jpg','D_4_small.jpg','D_5_small.png','D_move_small.png'
	]                
def evaluate_image_dir(model, files):
		ti = 0.0
		for i in range(len(files)):
			ti+=evaluate_model(model,files[i])
		print('FPS:{:.4f}, {} frames can be processed per second.'.format((len(files)/ti),int(len(files)/ti)))
		
evaluate_image_dir(z,files)



#tar -xf name_of_file 
