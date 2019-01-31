import os
from pathlib import Path
from flask import Flask, render_template, request, send_file
from flask import jsonify
import gatys
import gan
import database
import utils
web_app = Flask(__name__)
web_app.config["CACHE_TYPE"] = "null"


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# This provides a web interface to visualise the training and demonstration of 
#       different CNN approaches to texture/style synthesis
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#Return index page
@web_app.route("/")
def index():
    return render_template('index.html')

#Generate an image using gatys method
@web_app.route('/start_gatys')
def start_gatys():
    source_image = 'textures/' + request.args.get('source') + '.jpg'
    try:
        gatys.generate_texture(source_image, 'gatys', 0.8, 400, tileable=False, save_intermediates=True)
        success=True
    except:
        print('Exception in gatys')
        success=False
    finally:
        return jsonify(success)

#Generate an image using gan method
@web_app.route('/start_gan')
def start_gan():
    source_gan = request.args.get('source')
    try:
        gan.visualise(source_gan, 254, image_size=256)
        success=True
    except:
        print('Exception in gan')
        success=False
    finally:
        return jsonify(success)

#Return an image
@web_app.route('/image')
def get_image():
    target = request.args.get('target')
    alt = request.args.get('alt')
    my_file = Path(target)
    if my_file.exists():
        return send_file(target, mimetype='image/jpg')
    else:
        return send_file(alt, mimetype='image/jpg')

#Return progress of image generation
@web_app.route('/progress')
def gatys_progress():
    subject = request.args.get('subject')
    return jsonify(database.get_progress(subject))

#Return a list of images in the textures folder
@web_app.route('/get_source_images')
def get_src_images():
    source_image_list = []
    for file in os.listdir("textures"):
        if file.endswith(".jpg"):
            source_image_list.append(file)
    return jsonify(source_image_list)

if __name__ == "__main__":
    database.clear_db()
    database.init_db()
    database.populate_db()
    utils.crop_texture_images()
    web_app.run()
    #web_app.run(host='192.168.0.41', port='5000', debug=False)

