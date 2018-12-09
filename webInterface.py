from flask import Flask, flash, redirect, render_template, request, session, abort
from flask import send_file
from pathlib import Path
from flask import jsonify
import gatys
import GAN
import os
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
    return render_template(
        'index.html')

#Call gatys method 
@app.route('/start_gatys')
def start_gatys():
    if os.path.exists("temp/gatys.txt"):
        os.remove("temp/gatys.txt")
    source = request.args.get('source')
    target = request.args.get('target')
    learning_rate = request.args.get('learning_rate')
    iterations = request.args.get('iterations')
    tile = bool(request.args.get('tile'))
    print(tile)
    gatys.generate_texture(source, target, float(learning_rate), int(iterations), tileable=tile)
    return jsonify(success=True)

#Call GAN demo method
@app.route('/demo_GAN')
def demo_GAN():
    if os.path.exists("temp/GAN_demo.jpg"):
        os.remove("temp/GAN_demo.jpg")
    source = request.args.get('source')
    GAN.demonstrate_GAN(source)
    return jsonify(success=True)

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
    GAN.train_GAN(source, float(learning_rate), int(iterations), target, resume == 'true', generator)
    return jsonify(success=True)

#Return an image
@app.route('/image')
def gatys_image():
    target = request.args.get('target')
    alt = request.args.get('alt')
    my_file = Path(target)
    if my_file.exists():
        return send_file(target, mimetype='image/jpg')
    else:
        print(target)
        return send_file(alt, mimetype='image/jpg')

#Return progress of gatys texture generation
@app.route('/progress')
def gatys_progress():
    if os.path.exists("temp/gatys.txt"):
        f=open("temp/gatys.txt", "r")
        contents =f.read()
        f.close()
        return jsonify(len(contents))
    else:
        return jsonify(0)

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

#Return a list of images in the textures folder
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
    model_list = []
    #Get models
    for file in os.listdir("models"):
        if file.endswith("_gen.pt"):
            model_list.append(file) 
    return jsonify(model_list)

@app.route('/tile_image/<image>')
def hello(image=None):
    image = 'temp/' + image + '.jpg&r=' + str(random.randint(0,999999))
    return render_template('tile.html', image=image)

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

if __name__ == "__main__":
    app.run()