# 211127
# original code from Microsoft Corporation
# original code from https://recipe.narekomu-ai.com/2017/10/chainer_web_demo_2/

from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
import json
import os
import argparse

MODEL_DIR = os.path.join(os.path.dirname(__file__))


class Model(object):
    def __init__(self, model_dir=MODEL_DIR):
        # make sure our exported SavedModel folder exists
        model_path = os.path.realpath(model_dir)
        if not os.path.exists(model_path):
            raise ValueError(
                f"Exported model folder doesn't exist {model_dir}")
        self.model_path = model_path

        # load our signature json file, this shows us the model inputs and outputs
        # you should open this file and take a look at the inputs/outputs to see their data types, shapes, and names
        with open(os.path.join(model_path, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.inputs = self.signature.get("inputs")
        self.outputs = self.signature.get("outputs")

        # placeholder for the tensorflow session
        self.session = None

    def load(self):
        self.cleanup()
        # create a new tensorflow session
        self.session = tf.compat.v1.Session(graph=tf.Graph())
        # load our model into the session
        tf.compat.v1.saved_model.loader.load(
            sess=self.session, tags=self.signature.get("tags"), export_dir=self.model_path)

    def predict(self, image: Image.Image):
        # load the model if we don't have a session
        if self.session is None:
            self.load()
        # get the image width and height
        width, height = image.size
        # center crop image (you can substitute any other method to make a square image, such as just resizing or padding edges with 0)
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            # Crop the center of the image
            image = image.crop((left, top, right, bottom))
        # now the image is square, resize it to be the right shape for the model input
        if "Image" not in self.inputs:
            raise ValueError(
                "Couldn't find Image in model inputs - please report issue to Lobe!")
        input_width, input_height = self.inputs["Image"]["shape"][1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))
        image = np.asarray(image) / 255.0
        feed_dict = {self.inputs["Image"]["name"]: [image]}

        fetches = [(key, output["name"])
                   for key, output in self.outputs.items()]

        outputs = self.session.run(
            fetches=[name for _, name in fetches], feed_dict=feed_dict)
        results = {}
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        return results

    def cleanup(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def __del__(self):
        self.cleanup()


# Flaskオブジェクトの生成
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        # アプロードされたファイルを保存する
        f = request.files['file']
        filepath = "./static/" + datetime.now().strftime("%Y%m%d%H%M%S") + ".jpg"
        f.save(filepath)

        if os.path.isfile(filepath):
            image = Image.open(filepath)
            # convert to rgb image if this isn't one
            if image.mode != "RGB":
                image = image.convert("RGB")

            # モデルを使って判定する
            model = Model()
            model.load()

            outputs = model.predict(image)
            predict = f"{outputs}"
            # print(f"Predicted: {outputs}")
        else:
            predict = f"Couldn't find image file {filepath}"
            # print(f"Couldn't find image file {args.image}")

        return render_template('index.html', filepath=filepath, predict=predict)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
