#!/usr/bin/python

'''Trained model obstacle recognizer (TROM) software'''
import numpy as np
import os, sys
from cntk.ops.functions import load_model
from PIL import Image 
import scipy.misc
import cv2
import time
import argparse




from IPython.display import Image as display 
from resizeimage import resizeimage


size = 32

label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#z = load_model('./VGG Models/pred_vgg9_300_400.dnn')
#z = load_model('./VGG Models/pred_vgg9.dnn')
z = load_model('./pred_vgg9_300_400.dnn')

arg_parser = argparse.ArgumentParser()	
arg_parser.add_argument("-d", "--image_dir", type=str, default="./images", required=True,
	help="the directory to get test images")

args = vars(arg_parser.parse_args())
        
images_dir = f"{args['image_dir']}"

print(f"Image Directory: {images_dir}")

def first_image(direc):
	for file in direc:
		if file.find(".") >=0 and file.find(".py") <0:
			return file
	return None
	
def change_to_jpeg(imgurl):
	im = Image.open(imgurl)
	bg = Image.new("RGB", (size,size), (255,255,255))
	bg.paste(im,mask=im.split()[3])
	#n_name = imgurl[:-4]
	#full_name = f"{n_name}"+".jpg"
	#print(f"N_NAME: {full_name}")
	#bg.save(full_name)
	print("Changed to JPEG")
	#return full_name
	return bg

	
def paste_jpeg(img):
	bg = Image.new("RGB", (size,size), (255,255,255))
	bg.paste(img,mask=im.split()[3])
	return bg
	
def resize_image(im):
	oriimg = cv2.imread(im)
	newimg = oriimg
	im_shape = oriimg.shape
	if im_shape[2] == 4: newimg = paste_jpeg(newimg)
	if im_shape != (size,size,3):
		print('RESIZED, Shape: {}'.format(im_shape))
		newimg = cv2.resize(newimg,(32,32))
	return newimg

def whatIsMyName(name):
	label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	im_names = ["ai", "au", "bi", "ca", "de", "do", "fr", "ho", "sh", "tr"]
	if name[0] is "_": return "Car"
	#print("WIMN: {}".format(name[:2]))
	for x in range(len(im_names)):
		if name[:2].lower() == im_names[x] and name[1] is not "_":
			n_name = label_lookup[x][0].upper()+label_lookup[x][1:]
			return n_name
	return name
	
def evaluate_model(pred_op, image_path):
	n_name = image_path[len(images_dir)+1:]
	m_name=whatIsMyName(n_name)
	#print("Image {}".format(n_name))
	print("Image {}".format(whatIsMyName(m_name)))
	label_lookup = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
	im_names = ["ai", "au", "bi", "ca", "de", "do", "fr", "ho", "sh", "tr"]
	st_time = time.clock()
	bgr_im = resize_image(image_path)
	bgr_image = np.asarray(bgr_im, dtype=np.float32) - 127.5
	bgr_image = bgr_image[..., [2, 1, 0]]
	pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))
	result = np.squeeze(pred_op.eval({pred_op.arguments[0]:[pic]}))
	names = ['Automobile','Airplane','Dog','Horse','Car','Ship','Truck','Bird']
	test_name = '_PDHCSTB'
	img_name = image_path[len(images_dir)+1:]
	name = names[test_name.find(img_name[0])]
	if name[0] is "D":
		if n_name[:2] == "Do" or n_name[1] == "_": name = "Dog"
		else: name = "Deer"
    # Return top 5 results:
	top_count = 5
	result_indices = (-np.array(result)).argsort()[:top_count]
	end_time = time.clock()
	print("Top 5 predictions:")
	vowels ='oiuea'
	for i in range(top_count):
		label = label_lookup[result_indices[i]]
		conf = result[result_indices[i]] * 100
		article = 'an'
		if vowels.find(label_lookup[result_indices[i]][:1]) == -1:
			article ='a'
		if  result[result_indices[i]] * 100 >= 99.5:
			print("\tConfident {:s} {:8s}, confidence: {:.2f}%".format(article,label, conf))
		elif result[result_indices[i]] * 100 > 90.0:
			print("\tVery likely {:s} {:7s}, confidence: {:.2f}%".format(article,label, conf))
		else:
			print("\tLabel: {:14s}, confidence: {:.2f}%".format(label, conf))
		if name is "Car": name = "Automobile"
	print('Compute time was {:.10f}, and image was {:s}, name:{:s}'.format(end_time-st_time,m_name,n_name))
	return (end_time-st_time)

fir_ima = first_image(os.listdir(images_dir))
rgb_image = np.asarray(resize_image(images_dir+f"/{fir_ima}"), dtype=np.float32) - 128
bgr_image = rgb_image[..., [2, 1, 0]]
pic = np.ascontiguousarray(np.rollaxis(bgr_image, 2))

predictions = np.squeeze(z.eval({z.arguments[0]:[pic]}))
top_class = np.argmax(predictions)



def evaluate_image_dir(model, images_dir):
		ti = 0.0		
		if not os.path.exists(images_dir):
			print(f"No such dir: {images_dir}")
			sys.exit(1)
		dirs = os.listdir(images_dir)
		print(f"Image Dir Content: {dirs}")
		for file in dirs:
			if file.find(".") >=0 and file.find(".py") <0 and file.find(".dnn") < 0 :
				ti+=evaluate_model(model,f"{images_dir}/"+file)
		print('FPS:{:.4f}, {} frames can be processed per second.'.format((len(dirs)/ti),int(len(dirs)/ti)))
		
evaluate_image_dir(z,images_dir)
