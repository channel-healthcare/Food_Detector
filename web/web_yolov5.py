import argparse
import io

from flask import Flask, request, jsonify
import os
import time
import datetime
from datetime import timedelta

from detect_utils import Yolov5_Detector
from werkzeug.utils import secure_filename
from base64 import encodebytes
from PIL import Image
import pandas as pd
import json
from preprocess.micells import meta_header

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def Init(device='cuda:0', data_path=None, weight_path=None):
    startTime = time.time()

    cur_dir = os.getcwd()
    global detector, data_folder, meta_table, nutrients_list
    if data_path == None or weight_path == None:
        data_folder = os.path.join(cur_dir, '../web_data')
        weights = os.path.join(cur_dir, '../weights/best.pt')
    else:
        data_folder = data_path
        weights = weight_path

    detector = Yolov5_Detector()
    detector.load_model(weights=weights, device=device)

    table_path = 'data/FoodMetaTable.csv'
    jsons_path = 'data/meta_data'
    nutrients_names = 'data/nutrients_meta.json'
    meta_table, nutrients_list = load_meta_from_json(jsons_path, table_path, nutrients_names, meta_header)

    endTime = time.time()
    print(f'Yolov5 Loaded. Elapsed Time = {timedelta(seconds=endTime - startTime)}')


def load_meta_from_json(jsons_path, table_path, nutrients_names, meta_header):
    df = pd.read_csv(table_path, names=meta_header, header=0)
    dic_replace_name = dict(zip(df.food_name, df.train_name))
    with open(nutrients_names, 'r', encoding='utf-8') as f:
        nutrients_json = json.load(f)

    nutrients_list = list(nutrients_json.keys())

    json_list = os.listdir(jsons_path)
    dic_meta = {}
    for jsonfile in json_list:
        with open(os.path.join(jsons_path, jsonfile), 'r', encoding='utf-8') as f:
            json_info = json.load(f)

        class_name = json_info['Name']
        new_name = dic_replace_name.get(class_name)
        if new_name is not None:
            class_name = new_name

        ingre_info = [json_info[ingre] for ingre in nutrients_list]
        dic_meta[class_name] = ingre_info

    return dic_meta, nutrients_list


def get_nutrients_info(food_name):
    nutrients_info = meta_table.get(food_name)
    if nutrients_info is None:
        return None
    else:
        return dict(zip(nutrients_list, nutrients_info))




def image_path(filename):
    now = datetime.datetime.now()
    src_dir = os.path.join(data_folder, f'{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}')
    if not os.path.exists(src_dir):
        os.mkdir(src_dir)

    save_dir = os.path.join(src_dir, 'detect_result')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    file_path = os.path.join(src_dir, filename)
    return file_path, src_dir, save_dir


def encoded_image(file_path):
    pil_img = Image.open(file_path, mode='r')
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG')
    encode_img = encodebytes(byte_arr.getvalue()).decode('ascii')
    return encode_img


app = Flask('Food Images Detector')
app.secret_key = 'channel.healthcare'
app.config['JSON_AS_ASCII'] = False
current_version = 0.3
response_template = {'version': current_version, 'error': 0, 'message': '', 'image': None}


@app.route('/', methods=['POST', 'GET'])
def home():
    print(f'Remote Host: {request.remote_addr}')
    return 'Food Images Detecting Server'


def msg_error(msg):
    response = response_template
    response['error'] = 1
    response['message'] = msg
    response['image'] = None
    response['crops'] = None
    print(f'Error = {msg}')
    return response


def build_json(save_dir, file_name):
    response = response_template
    crop_path = os.path.join(save_dir, 'crops')
    if not os.path.exists(crop_path):
        response['error'] = 1
        response['message'] = f'cannot detect food from file = {file_name}'
        return response

    result_file = os.path.join(save_dir, file_name)
    encoded_img = encoded_image(result_file)
    response['error'] = 0
    response['message'] = ''
    response['image'] = encoded_img

    crop_list = []
    for r, d, f in os.walk(crop_path):
        if len(d) == 0:
            class_name = os.path.split(r)[1]
            for image_file in f:
                result_file = os.path.join(r, image_file)
                encoded_img = encoded_image(result_file)
                crop_info = {'class': class_name, 'nutrients': get_nutrients_info(class_name), 'image': encoded_img}
                crop_list.append(crop_info)
    response['crops'] = crop_list
    return response


@app.route('/detect', methods=['POST'])
def detect():
    if not (request.method == 'POST' and 'photo' in request.files):
        return jsonify(msg_error('request error "photo" dost not exist in request'))
    elif request.form['version'] != str(current_version):
        return jsonify(msg_error(f'version number must be {current_version}'))
    else:
        startTime = time.time()
        file = request.files['photo']
        if file.filename == '':
            return jsonify(msg_error('no images'))

        filename = secure_filename(file.filename)
        file_path, src_dir, save_dir = image_path(filename)
        file.save(file_path)
        label = True if request.form['label'] == '1' else False
        confidence = True if request.form['confidence'] == '1' else False

        detector.detect(source=src_dir, save_dir=save_dir, save_crop=True, save_txt=False,
                        save_img=True, view_img=False, show_label=label, show_conf=confidence)
        response = build_json(save_dir, filename)
        endTime = time.time()
        print(f'Detecting Time = {timedelta(seconds=endTime - startTime)}')

        return jsonify(response)


def get_params():
    parser = argparse.ArgumentParser(description='receive parameters')
    parser.add_argument('--host', type=str, required=True, help='type: 0.0.0.0 or ip address')
    parser.add_argument('--port', type=int, required=True, help='type: network port')
    parser.add_argument('--debug', type=str2bool, required=True, help='type: True or False')
    parser.add_argument('--use_reloader', type=str2bool, required=True, help='type: True or False')
    parser.add_argument('--device', type=str, required=True, help='type: cuda:0 or cpu')
    parser.add_argument('--data_folder', type=str, required=True, help='type: relative path')
    parser.add_argument('--weights', type=str, required=True, help='type: relative path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    params = get_params()
    Init(params.device, params.data_folder, params.weights)
    print(f'========================Server Start device = {params.device}==============================')
    app.run(host=params.host, port=params.port, debug=params.debug, use_reloader=params.use_reloader)
