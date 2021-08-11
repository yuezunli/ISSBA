"""
The original code is from StegaStamp: 
Invisible Hyperlinks in Physical Photographs, 
Matthew Tancik, Ben Mildenhall, Ren Ng 
University of California, Berkeley, CVPR2020
More details can be found here: https://github.com/tancik/StegaStamp 
"""
import bchlib
import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import signature_constants
import argparse


parser = argparse.ArgumentParser(description='Generate sample-specific triggers')
parser.add_argument('--model_path', type=str, default='ckpt/encoder_imagenet')
parser.add_argument('--image_path', type=str, default='data/imagenet/org/n01770393_12386.JPEG')
parser.add_argument('--out_dir', type=str, default='data/imagenet/bd/')
parser.add_argument('--secret', type=str, default='a')
parser.add_argument('--secret_size', type=int, default=100)
args = parser.parse_args()


model_path = args.model_path
image_path = args.image_path
out_dir = args.out_dir
secret = args.secret # lenght of secret less than 7
secret_size = args.secret_size


sess = tf.InteractiveSession(graph=tf.Graph())

model = tf.saved_model.loader.load(sess, [tag_constants.SERVING], model_path)

input_secret_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['secret'].name
input_image_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].inputs['image'].name
input_secret = tf.get_default_graph().get_tensor_by_name(input_secret_name)
input_image = tf.get_default_graph().get_tensor_by_name(input_image_name)

output_stegastamp_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['stegastamp'].name
output_residual_name = model.signature_def[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY].outputs['residual'].name
output_stegastamp = tf.get_default_graph().get_tensor_by_name(output_stegastamp_name)
output_residual = tf.get_default_graph().get_tensor_by_name(output_residual_name)

width = 224
height = 224

BCH_POLYNOMIAL = 137
BCH_BITS = 5
bch = bchlib.BCH(BCH_POLYNOMIAL, BCH_BITS)

data = bytearray(secret + ' '*(7-len(secret)), 'utf-8')
ecc = bch.encode(data)
packet = data + ecc

packet_binary = ''.join(format(x, '08b') for x in packet)
secret = [int(x) for x in packet_binary]
secret.extend([0, 0, 0, 0])

image = Image.open(image_path)
image = np.array(image, dtype=np.float32) / 255.

feed_dict = {
    input_secret:[secret],
    input_image:[image]
    }

hidden_img, residual = sess.run([output_stegastamp, output_residual],feed_dict=feed_dict)

hidden_img = (hidden_img[0] * 255).astype(np.uint8)
residual = residual[0] + .5  # For visualization
residual = (residual * 255).astype(np.uint8)

name = os.path.basename(image_path).split('.')[0]

im = Image.fromarray(np.array(hidden_img))
im.save(out_dir + '/' + name + '_hidden.png')
im = Image.fromarray(np.squeeze(residual))
im.save(out_dir + '/' + name + '_residual.png')
