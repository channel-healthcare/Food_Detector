import os
import subprocess
import sys
import time
import zipfile
from datetime import timedelta

import yaml
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial
import csv
import pandas as pd
import parmap
import json
import shutil
from zipfile import ZipFile

import numpy as np
import matplotlib.pyplot as plt

# os.system('chcp 65001')

def build_summary(img_path, out_file):
    out_dict = dict()
    food_info = dict()
    disagree_list = list()
    img_total_cnt, json_total_cnt = 0, 0
    json_folder = os.path.join(os.path.split(img_path)[0], 'jsons')
    json_folder_list = os.listdir(json_folder)
    json_folder_list = [file.replace(' json', '') for file in json_folder_list]
    for r, d, f in tqdm(list(os.walk(img_path)), desc='building summary images and json'):
        if r == img_path:
            image_folder_list = d
        if len(d) == 0:
            class_name = os.path.split(r)[1]
            json_folder_name = os.path.join(json_folder, f'{class_name} json')
            if os.path.exists(json_folder_name):
                json_cnt = len(os.listdir(json_folder_name))
            else:
                json_cnt = 0
            img_cnt = len(f)
            food_info[class_name] = {'images': img_cnt, 'jsons': json_cnt}
            if img_cnt != json_cnt:
                disagree_list.append({'name': class_name, 'info': {'images': img_cnt, 'jsons': json_cnt}})
            img_total_cnt += img_cnt
            json_total_cnt += json_cnt

    out_dict['food folder information'] = food_info
    out_dict['disagree folder names'] = disagree_list
    out_dict['only image exist folder'] = list(set(image_folder_list) - set(json_folder_list))
    out_dict['only json exist folder'] = list(set(json_folder_list) - set(image_folder_list))
    out_dict['disagree folder count'] = len(disagree_list)
    out_dict['total image folder count'] = len(image_folder_list)
    out_dict['total json folder count'] = len(json_folder_list)
    out_dict['total image count'] = img_total_cnt
    out_dict['total json count'] = json_total_cnt
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)

def fix_json_foldername(root, out_file):
    out_dict = dict()
    fault_list = list()
    pure_list = list()
    fine_list = list()
    for r, d, f in tqdm(list(os.walk(root)), desc='fix json folder names'):
        if len(d) ==0:
            folder_name = os.path.split(r)[1]
            if ' json' in folder_name:
                fine_list.append(folder_name)
            elif '.json' in folder_name:
                fault_list.append(folder_name)
            else:
                pure_list.append(folder_name)

    out_dict['fine counts'] = len(fine_list)
    out_dict['.json counts'] = len(fault_list)
    out_dict['pure counts'] = len(pure_list)
    out_dict['fine list'] = fine_list
    out_dict['.json list'] = fault_list
    out_dict['pure list'] = pure_list

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2, ensure_ascii=False)

    os.chdir(root)
    for file in fault_list:
        shutil.move(file, f'{file.replace(".", " ")}')

    for file in pure_list:
        shutil.move(file, f'{file} json')


zip = 'C:/Program Files/7-Zip/7z.exe'
# zip = '7z.exe'
def zip_folders(folder_list, src_path, tar_path):
    os.chdir(src_path)
    command = f'"{zip}" a' \
              f' {os.path.join(tar_path, str(folder_list["index"]))}.zip'
              # f' ../json_zip/{str(folder_list["index"])}.zip'

    for folder in folder_list['list']:
        command += f' ./"{folder}"'
    command += ' > nul'
    print(command)
    os.system(command)

def upzip_files(filelist, src_path, tar_path):
    pid = multiprocessing.current_process()
    for afile in tqdm(filelist, desc=f'unzip folders, pid = {pid}'):
        zip_file = zipfile.ZipFile(os.path.join(src_path, afile))
        zip_file.extractall(tar_path)
        zip_file.close()

def upzip_all(src_path, tar_path, job_count=1):
    if job_count == 1:
        zip_list = os.listdir(src_path)
        for afile in tqdm(zip_list):
            zip_file = zipfile.ZipFile(os.path.join(src_path, afile))
            zip_file.extractall(tar_path)
            zip_file.close()
    else:
        zip_list = os.listdir(src_path)
        job_number = job_count
        total = len(zip_list)
        chunk_size = total // job_number
        batches = chunks_list(zip_list, chunk_size)
        parmap.map(upzip_files, batches, src_path=src_path, tar_path=tar_path, pm_pbar=True, pm_processes=job_number)


def zipfile_folders(folder_list, tar_path):
    zipf = zipfile.ZipFile(f'{os.path.join(tar_path, str(folder_list["index"]))}.zip', 'w', zipfile.ZIP_DEFLATED)
    for folder in tqdm(folder_list['list'], desc=f'== zip index = {folder_list["index"]}'):
        for r, d, f in os.walk(folder):
            for file in f:
            # for file in tqdm(f, desc=f'food dir = {os.path.split(r)[1]}'):
                zipf.write(os.path.join(r, file),
                           os.path.relpath(os.path.join(r, file), os.path.join(folder, '..')))
    zipf.close()

def zip_json_all(src_path, tar_path, zip_count=1):
    if zip_count == 1:
        os.chdir(src_path)
        command = f'"{zip}" a {os.path.join(tar_path, os.path.split(src_path)[1])}.zip '
        print(command)
        os.system(command)
    else:
        dir_list = []
        for r, d, f in os.walk(src_path):
            if r == src_path:
                dir_list = d
                break

        os.chdir(src_path)
        job_number = zip_count
        total = len(dir_list)
        chunk_size = total // job_number
        chunk_list = chunks_list(dir_list, chunk_size)
        batches = [{'index': index, 'list': alist} for index, alist in enumerate(chunk_list)]
        parmap.map(zipfile_folders, batches, tar_path=tar_path, pm_pbar=True, pm_processes=job_number)

def zip_folder2zip(folder_list, tar_path):
    pid = multiprocessing.current_process()
    for folder in tqdm(folder_list, desc=f'== zip index = {pid}'):
        zipf = zipfile.ZipFile(f'{os.path.join(tar_path, folder)}.zip', 'w', zipfile.ZIP_DEFLATED)
        abs_folder = os.path.join(folder)
        for r, d, f in os.walk(abs_folder):
            for file in f:
                zipf.write(os.path.join(r, file),
                           os.path.relpath(os.path.join(r, file), os.path.join(folder, '..')))
        zipf.close()


def zip_image_all(src_path, tar_path, job_count=1):
    if job_count == 1:
        pass
    else:
        os.chdir(src_path)
        dir_list = os.listdir(src_path)
        job_number = job_count
        total = len(dir_list)
        chunk_size = total // job_number
        batches = chunks_list(dir_list, chunk_size)
        parmap.map(zip_folder2zip, batches, tar_path=tar_path, pm_pbar=True, pm_processes=job_number)


def chunks_list(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def save_filespec(src_path, out_file):
    file_list = os.listdir(src_path)
    yaml_out = {}
    yaml_out['file count'] = len(file_list)
    yaml_out['zip files'] = file_list
    with open(out_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_out, f, default_flow_style=False, allow_unicode=True)


def sort_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    lost_zips = yaml_data['lost zips']
    lost_zips.sort()
    yaml_data['lost zips'] = lost_zips

    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True)


def move_files(src_path, tar_path, yaml_file):
    with open(yaml_file, 'r', encoding='utf-8') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    lost_zips = yaml_data['lost zips']
    for file in lost_zips:
        shutil.move(os.path.join(src_path, file), tar_path)


def copy_files(file_list, tar_path):
    pid = multiprocessing.current_process()
    for file in tqdm(file_list, desc=f'copy files. pid = {pid.name}'):
        shutil.copy(file, tar_path)


def copy_files_all(src_path, tar_path, job_count):
    os.chdir(src_path)
    file_list = os.listdir(src_path)
    job_number = multiprocessing.cpu_count() if job_count == -1 else job_count
    total = len(file_list)
    chunk_size = total // job_number
    batches = chunks_list(file_list, chunk_size)
    parmap.map(copy_files, batches, tar_path=tar_path, pm_pbar=True, pm_processes=job_number)

def copy_files_diff(src_path, tar_path, diff_file, job_count):
    os.chdir(src_path)
    df = pd.read_csv(diff_file, names=['diff'])
    file_list = df['diff'].tolist()
    job_number = multiprocessing.cpu_count() if job_count == -1 else job_count
    total = len(file_list)
    chunk_size = total // job_number
    batches = chunks_list(file_list, chunk_size)
    parmap.map(copy_files, batches, tar_path=tar_path, pm_pbar=True, pm_processes=job_number)


column = ['epoch', 'train/box_loss', 'train/obj_loss', 'train/cls_loss', 'metrics/precision', 'metrics/recall',
          'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 'val/cls_loss', 'x/lr0', 'x/lr1', 'x/lr2']

def predict_mAP(table_path, epoch):
    df = pd.read_csv(table_path, names=column, usecols=['epoch', 'metrics/mAP_0.5'], header=0)
    df.columns = ['epoch', 'mAP']
    slope = (df.loc[69, 'mAP'] - df.loc[14, 'mAP'])/(69 - 10)
    return slope * (epoch - 69) + df.loc[69, 'mAP']

def draw_lr_curve(From, To, saveTo):
    x = np.arange(From, To)
    y = (1-x/To) * (1.0 - 0.01) + 0.1
    plt.plot(x, y)
    plt.show()
    plt.savefig(saveTo)


def diff_folders(src_path, tar_path, none_out, size_out):
    src_list = os.listdir(src_path)
    tar_list = os.listdir(tar_path)
    none_file, size_file = [], []
    for src_file in tqdm(src_list, desc=f'diff folders'):
        if src_file not in tar_list:
            none_file.append(src_file)
            continue

        src_f = os.path.join(src_path, src_file)
        tar_f = os.path.join(tar_path, src_file)
        if os.path.getsize(src_f) != os.path.getsize(tar_f):
            size_file.append(src_file)

    if len(none_file) ==  0 and len(size_file) == 0:
        print('There is nothing different')
        return

    df = pd.DataFrame(none_file, columns=['none file'])
    df.to_csv(none_out, index=False, encoding='utf-8')

    df = pd.DataFrame(size_file, columns=['size file'])
    df.to_csv(size_out, index=False, encoding='utf-8')


def save_filename(tar_path, out_file):
    file_list = os.listdir(tar_path)
    df = pd.DataFrame(file_list)
    df.to_csv(out_file, index=False, encoding='utf-8')


def diff_zips(in_file, out_file, tar_path):
    df = pd.read_csv(in_file, names=['name'], header=0)
    src_list = df.name.tolist()
    tar_list = os.listdir(tar_path)
    diff = set(src_list) - set(tar_list)
    df = pd.DataFrame(diff)
    df.to_csv(out_file, index=False, encoding='utf-8')


def move_folder_up(src_path, tar_path):
    if not os.path.exists(tar_path): os.mkdir(tar_path)
    root_list = os.listdir(src_path)
    for root in root_list:
        dir_list = os.listdir(os.path.join(src_path, root))
        dir = os.path.join(src_path, root, dir_list[0])
        shutil.move(dir, tar_path)


def move_txt_to(src_path, tar_path):
    if not os.path.exists(tar_path): os.mkdir(tar_path)
    root_list = os.listdir(src_path)
    txt_list = [file for file in root_list if 'txt' in file]
    for txt in txt_list:
        txt_path = os.path.join(src_path, txt)
        shutil.move(txt_path, tar_path)


if __name__ == '__main__':
    src_path = sys.argv[1]
    tar_path = sys.argv[2]
    print(f'=============== move txt from {src_path} to {tar_path} =====================')
    move_txt_to(src_path, tar_path)

    # # tar_path = '/home2/channelbiome/dev_gdrive/NIA_2021/images'
    # # in_file = '/home2/channelbiome/NIA_Zips/zipfiles.csv'
    # # out_file = '/home2/channelbiome/NIA_Zips/zipfiles_diff.csv'
    # tar_path = sys.argv[1]
    # in_file = sys.argv[2]
    # out_file = sys.argv[3]
    # print(f'=============== save file name  {tar_path} =====================')
    # diff_zips(in_file, out_file, tar_path)

    # tar_path = '/home2/channelbiome/dev_gdrive/NIA_2021/images'
    # out_file = '/home2/channelbiome/NIA_Zips/zipfiles.csv'
    # print(f'=============== save file name  {tar_path} =====================')
    # save_filename(tar_path, out_file)

    # startTime = time.time()
    # # src_path = '/home2/channelbiome/NIA_Data/2021/test'
    # # tar_path = '/home2/channelbiome/NIA_Data/2021/test_moved'
    # src_path = sys.argv[1]
    # tar_path = sys.argv[2]
    # print(f'=============== move folder up {src_path} to {tar_path} =====================')
    # move_folder_up(src_path, tar_path)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime-startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/NIA_Zips/NIA_2021_images'
    # tar_path = '/home2/channelbiome/googleDrive/nia2021/images'
    # diff_file = '/home2/channelbiome/googleDrive/diff_files.csv'
    # print(f'=============== start copy diff files {src_path} to {tar_path} =====================')
    # copy_files_diff(src_path, tar_path, diff_file, job_count=20)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime-startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/NIA_Zips/NIA_2021_images'
    # tar_path = '/home2/channelbiome/dev_gdrive/NIA_2021/images'
    # none_out = '/home2/channelbiome/NIA_Zips/none_file.csv'
    # size_out = '/home2/channelbiome/NIA_Zips/size_file.csv'
    # print(f'=============== start diff folders {src_path} to {tar_path} =====================')
    # diff_folders(src_path, tar_path, none_out, size_out)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime-startTime)}')

    # startTime = time.time()
    # src_path = sys.argv[1]
    # tar_path = sys.argv[2]
    # job_count = int(sys.argv[3])
    # a = f'/home2/channelbiome/NIA_Zips/'
    # b = f'/home2/channelbiome/dev_gdrive/NIA_2021/jsons'
    # print(f'=============== start copy files {src_path} to {tar_path} =====================')
    # copy_files_all(src_path, tar_path, job_count=job_count)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime-startTime)}')

    # From = 0
    # To = 200
    # saveTo = 'data/lr_curve.png'
    # draw_lr_curve(From, To, saveTo)

    # table_path = 'data/results.csv'
    # epoch = 100
    # result = predict_mAP(table_path, epoch)
    # print(f'e = {epoch}, mAP = {result}')
    #
    # epoch = 200
    # result = predict_mAP(table_path, epoch)
    # print(f'e = {epoch}, mAP = {result}')
    #
    # epoch = 300
    # result = predict_mAP(table_path, epoch)
    # print(f'e = {epoch}, mAP = {result}')


    # startTime = time.time()
    # img_path = "I:/NIA2021FoodData/after_rearrange/images"
    # out_file = 'I:/NIA2021FoodData/after_rearrange/summary.json'
    # print(f'===============start build summary=====================')
    # build_summary(img_path, out_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime-startTime)}')

    # startTime = time.time()
    # json_path = "I:/NIA2021FoodData/after_rearrange/jsons"
    # out_file = 'I:/NIA2021FoodData/after_rearrange/json_folder_info.json'
    # print(f'=============== fix json folder names =====================')
    # fix_json_foldername(json_path, out_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime-startTime)}')

    # startTime = time.time()
    # src_path = "I:/NIA2021FoodData/after_rearrange/jsons"
    # tar_path = 'I:/NIA2021FoodData/after_rearrange/jsons_zip'
    # # src_path = "I:/NIA2021FoodData/after_rearrange/json_test"
    # # tar_path = 'I:/NIA2021FoodData/after_rearrange/js_zip'
    # zip_json_all(src_path, tar_path, zip_count=32)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')


    # startTime = time.time()
    # src_path = "/home2/channelbiome/NIA_Zips/test_unzip"
    # tar_path = "/home2/channelbiome/NIA_Zips/test_unzip_target"
    # upzip_all(src_path, tar_path, job_count=2)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # # src_path = "/home2/channelbiome/NIA_Data/2021/zip_images"
    # # tar_path = "/home2/channelbiome/NIA_Data/2021/zip_images_target"
    # src_path = "/home2/channelbiome/NIA_Data/2021/NIA_2021_Txt"
    # tar_path = "/home2/channelbiome/NIA_Zips/NIA_2021_labels"
    # zip_image_all(src_path, tar_path, job_count=32)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # src_path = 'I:/NIA2021FoodData/after_rearrange/images_zip'
    # out_file = 'I:/NIA2021FoodData/after_rearrange/images_zip_info.yaml'
    # save_filespec(src_path, out_file)

    # file_path = 'I:/NIA2021FoodData/after_rearrange/lost_zip_info.yaml'
    # sort_list(file_path)

    # startTime = time.time()
    # src_path = 'I:/NIA2021FoodData/after_rearrange/images_zip'
    # tar_path = "I:/NIA2021FoodData/after_rearrange/lost_zip"
    # yaml_file = 'I:/NIA2021FoodData/after_rearrange/lost_zip_info.yaml'
    # move_files(src_path, tar_path, yaml_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')
