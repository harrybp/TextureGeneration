from flask import Flask, flash, redirect, render_template, request, session, abort
from flask import send_file
from pathlib import Path
from flask import jsonify
import json
import gatys
import PSGAN
import os
import utils
import random
app = Flask(__name__)
app.config["CACHE_TYPE"] = "null"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This provides a web interface to visualise the training and demonstration of 
#       different CNN approaches to texture/style synthesis
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


#Return index page
@app.route("/")
def index():
    return render_template('index.html')

###############################################################################
#   Generate an image using Gatys method
@app.route('/start_gatys')
def start_gatys():
    source_image = 'textures/' + request.args.get('source') + '.jpg'
    learning_rate = request.args.get('learning_rate')
    iterations = request.args.get('iterations')
    toggle_tile_image = bool(request.args.get('tile'))
    gatys.generate_texture(source_image, 'gatys', float(learning_rate), int(iterations), tileable=toggle_tile_image, save_intermediates=True)
    return jsonify(success=True)

###############################################################################
#   Generate an image using Gatys method
@app.route('/start_gan')
def start_gan():
    source_gan = request.args.get('source')
    print(source_gan)
    PSGAN.visualise(source_gan, 254, image_size=256)
    #toggle_tile_image = bool(request.args.get('tile'))
    #gatys.generate_texture(source_image, 'gatys', float(learning_rate), int(iterations), tileable=toggle_tile_image, save_intermediates=True)
    return jsonify(success=True)

###############################################################################
#   Return progress of image generation
@app.route('/progress_gan')
def gan_progress():
    return jsonify(utils.get_progress_gan())



###############################################################################
#   Return an image
@app.route('/image')
def get_image():
    target = request.args.get('target')
    alt = request.args.get('alt')
    my_file = Path(target)
    if my_file.exists():
        return send_file(target, mimetype='image/jpg')
    else:
        return send_file(alt, mimetype='image/jpg')

###############################################################################
#   Return progress of image generation
@app.route('/progress')
def gatys_progress():
    return jsonify(utils.get_progress())

###############################################################################
#   Return a list of images in the textures folder
@app.route('/get_source_images')
def get_src_images():
    source_image_list = []
    for file in os.listdir("textures"):
        if file.endswith(".jpg"):
            source_image_list.append(file)
    print(source_image_list)
    return jsonify(source_image_list)

#Return a list of generators in the models folder
@app.route('/get_generators')
def get_generators():
    model_list = utils.get_GANS()
    return jsonify(model_list)

@app.route('/tile_image/<image>')
def hello(image=None):
    image = 'temp/gatys/' + image + '.jpg&r=' + str(random.randint(0,999999))
    return render_template('tile.html', image=image)



if __name__ == "__main__":
    #app.run()
    app.run(host='192.168.0.41', port='5000', debug=False)



##################################################################
@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

#Call GAN train method
@app.route('/train_GAN')
def train_GAN():
    if os.path.exists("temp/GANs.txt"):
        os.remove("temp/GANs.txt")
    resume = request.args.get('resume')
    generator = request.args.get('generator')[:-7]
    print(generator)
    source = request.args.get('source')
    target = request.args.get('target')
    learning_rate = request.args.get('learning_rate')
    iterations = request.args.get('iterations')
    GAN.train_GAN(source, float(learning_rate), int(iterations), target, resume == 'true', generator, save_intermediates=True)
    return jsonify(success=True)

#Return progress of GAN training
@app.route('/GAN_progress')
def GAN_progress():
    if os.path.exists("temp/GANs.txt"):
        f=open("temp/GANs.txt", "r")
        contents =f.read()
        f.close()
        return jsonify(len(contents))
    else:
        return jsonify(0)

###############################################################################
#   Generate an image using pretrained SGAN
@app.route('/demo_GAN')
def demo_GAN():
    name = request.args.get('name')
    iterations = request.args.get('iters')
    source = 'models/' + name + '/' + iterations + '/generator.pt'
    PSGAN.demonstrate_GAN(source, image_size=512)
    return jsonify(success=True)

#if __name__ == '__main__':
 #   app.run(host='192.168.0.41', port='5000', debug=False)