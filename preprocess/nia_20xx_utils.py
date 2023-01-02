import json
import sys
import os
import time
import shutil
from datetime import timedelta

import pandas as pd
import parmap
import yaml
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial


def chunks_list(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def diff_zip_files(yaml_file, tar_path, out_file):
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    out_yaml = {}

    src_list = yaml_data['zip files']
    tar_list = os.listdir(tar_path)
    lost_zips = list(set(src_list) - set(tar_list))
    out_yaml['lost file count'] = len(lost_zips)
    out_yaml['lost zips'] = lost_zips

    with open(out_file, 'w', encoding='utf-8') as f:
        yaml.dump(out_yaml, f, default_style=False, allow_unicode=True)


def check_blank_folder(src_path, out_file):
    blank_list = []
    for r, d, f in tqdm(list(os.walk(src_path)), desc=f'check blank folder = {src_path}'):
        if r != src_path and len(f) == 0:
            blank_list.append(os.path.split(r)[1])

    yaml_data = {'blank folder count': len(blank_list), 'blank folder': blank_list}
    with open(out_file, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_style=False, allow_unicode=True)


def check_file_count(src_path, in_file, out_file):
    global folder_name_list
    with open(in_file, 'r') as f:
        compare_data = yaml.load(f, Loader=yaml.FullLoader)

    listdir = os.listdir(src_path)
    listdir.sort()

    compare_dict = compare_data['food folder information']
    disagree_dic = {}
    for r, d, f in tqdm(list(os.walk(src_path)), desc=f'check blank folder = {src_path}'):
        if r == src_path:
            folder_name_list = d
            folder_name_list.sort()
            continue

        try:
            root = str(r)
            food_name = os.path.split(root)[1]
            origin_count = compare_dict[food_name]['images']
            cur_count = len(f)
            if origin_count != cur_count:
                disagree_dic[food_name] = {'origin count': origin_count, 'current count': cur_count}
        except Exception as e:
            print(f'exception food = {food_name}, what = {e}')

    json_data = {'dismatch folder count': len(disagree_dic.items()), 'dismatch folder': disagree_dic}

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def compare_image_json_folder(json_path, image_path, in_file, out_file):
    with open(in_file, 'r') as f:
        compare_data = yaml.load(f, Loader=yaml.FullLoader)

    origin_names = compare_data['food folder information'].keys()

    json_names = os.listdir(json_path)
    image_names = os.listdir(image_path)
    info_dic = {}
    dismatch_list = []
    for folder_name in tqdm(image_names, desc='Compare image and json'):
        json_name = f'{folder_name} json'
        if json_name in json_names:
            class_name = folder_name
            image_count = len(os.listdir(os.path.join(image_path, folder_name)))
            json_count = len(os.listdir(os.path.join(json_path, json_name)))
            info_dic[class_name] = {'image count': image_count, 'json count': json_count}
        else:
            dismatch_list.append(folder_name)

    no_images = list(set(origin_names) - set(image_names))
    json_data = {'image folder count': len(image_names), 'json folder count': len(json_names),
                 'dismatch count': len(dismatch_list), 'dismatch info': dismatch_list, 'food img/json info': info_dic,
                 'no image count': len(no_images), 'image no exists': no_images}

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def build_file2folder_dic(json_folder, out_file):
    folder_list = os.listdir(json_folder)
    for folder in folder_list:
        file_list = os.listdir(os.path.join(json_folder, folder))
        if len(file_list) == 0:
            print(f'There is no file in {folder}')


def upzip_temp_folder(src_path, tar_path, json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    upzip_folders = json_data['dismatch folder']
    for k, v in tqdm(upzip_folders.items(), desc=f'unzip files in {src_path} to {tar_path}'):
    # upzip_folders = json_data['image no exists']
    # for k in tqdm(upzip_folders, desc=f'unzip files in {src_path} to {tar_path}'):
        # tar_dir =tar_path
        tar_dir = os.path.join(tar_path, k)
        if not os.path.exists(tar_dir):
            os.mkdir(tar_dir)
        command = f'unzip -O cp949 -qq {os.path.join(src_path, k)}.zip -d {tar_dir}'
        print(f'{command}')
        os.system(command)


def move_broken_folder(src_path, tar_path, json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # right_folers = json_data['dismatch folder']
    right_folers = json_data['image no exists']
    for r, d, f in tqdm(list(os.walk(src_path)), desc=f'move broken folder to right folder'):
        if len(d) == 0:
            parent, folder_name = os.path.split(r)
            right_folder = os.path.split(parent)[1]
            if not folder_name in right_folers:
                for file in f:
                    # shutil.move(f'"{os.path.join(r, file)}"', f'{tar_path}')
                    From = f'"{os.path.join(r, file)}"'
                    To = f'"{os.path.join(tar_path, right_folder)}"'
                    shutil.move(From, To)


def move_and_rename(src_path, tar_path):
    for r, d, f in tqdm(list(os.walk(src_path)), desc=f'move and rename'):
        if len(d) == 0:
            parent, folder_name = os.path.split(r)
            right_folder = os.path.split(parent)[1]
            command = f'mv {r} {tar_path}'
            os.system(command)
            command = f'mv {os.path.join(tar_path, folder_name)} {os.path.join(tar_path, right_folder)}'
            os.system(command)


def unzip_afile(filelist, src_path, tar_path):
    pid = multiprocessing.current_process()
    for file in tqdm(filelist, desc=f'upzip folders, pid = {pid.name}'):
        command = f'unzip -O cp949 -qq "{os.path.join(src_path, file)}" -d "{os.path.join(tar_path, file)}"'
        # command = f'unzip -O cp949 -qq "{os.path.join(src_path, file)}" -d {tar_path}'
        os.system(command)


def unzip_skipped(src_path, tar_path, json_file, cpu_count=1):
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    skipped_list = json_data['image no exists']

    job_number = cpu_count if cpu_count > 1 else multiprocessing.cpu_count()
    # job_number = 1
    total = len(skipped_list)
    chunk_size = total // job_number
    batches = chunks_list(skipped_list, chunk_size)
    pool = Pool(job_number)
    copy_func = partial(unzip_afile, src_path=src_path, tar_path=tar_path)
    pool.map(copy_func, batches)
    print('Pool Ready')
    pool.close()
    pool.join()
    print('Pool Done')

def unzip_rename(zip_list, src_path, tar_path):
    for azip in zip_list:
        tar_dir = os.path.join(tar_path, f'{azip.split("_")[0]}')
        # if not os.path.exists(tar_dir):
        #     os.mkdir(tar_dir)
        command = f'unzip -O cp949 -qq {os.path.join(src_path, azip)} -d {tar_dir}'
        os.system(command)


def unzip_rename_all(src_path, tar_path, job_count=1):
    zip_files = os.listdir(src_path)
    total = len(zip_files)
    chunk_size = total // job_count
    batches = chunks_list(zip_files, chunk_size)
    parmap.map(unzip_rename, batches, src_path=src_path, tar_path=tar_path, pm_pbar=True, pm_processes=job_count)


def move_files(folder_list, from_path, to_path):
    for folder in folder_list:
        tar_folder = os.path.join(to_path, folder)
        if os.path.exists(tar_folder):
            file_list = os.listdir(os.path.join(from_path, folder))
            for file in file_list:
                shutil.move(os.path.join(from_path, folder, file), tar_folder)
        else:
            shutil.move(os.path.join(from_path, folder), tar_folder)


def move_all(from_path, to_path, job_count=1):
    folder_list = os.listdir(from_path)
    total = len(folder_list)
    chunk_size = total // job_count
    batches = chunks_list(folder_list, chunk_size)
    parmap.map(move_files, batches, from_path=from_path, to_path=to_path, pm_pbar=True, pm_processes=job_count)


def summary_nia2022(src_path, csv_path, out_file):
    df = pd.read_csv(csv_path, names=['name', 'cnt'], header=0)
    cnt_dict = dict(zip(df.name, df.cnt))
    json_info, food_info, diff_info = dict(), dict(), dict()
    img_total, txt_total = 0, 0

    folder_list = os.listdir(src_path)
    for folder in folder_list:
        file_list = os.listdir(os.path.join(src_path, folder))
        img_count = 0
        for file in file_list:
            img_count = img_count + 1 if os.path.splitext(file)[1] == '.jpg' else img_count
        try:
            txt_count = len(file_list) - img_count
            img_total += img_count
            txt_total += txt_count
            food_info[folder] = {'image count': img_count, 'txt count': txt_count}
            if cnt_dict[folder] != img_count:
                diff_info[folder] = {'origin count': len(cnt_dict[folder]), 'current count': img_count}
        except Exception as e:
            print(f'exception = {e}')

    json_info['class count'] = len(folder_list)
    json_info['image count'] = img_total
    json_info['txt count'] = txt_total
    json_info['count mismatch'] = diff_info
    json_info['food info'] = food_info

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(json_info, f, indent=2, ensure_ascii=False)



if __name__ == '__main__':
    pass
    # startTime = time.time()
    # src_path = '/home2/channelbiome/NIA_Data/2022/unzip_221130'
    # csv_path = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed/food_name2tot_count.csv'
    # out_file = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed/nia2022_summary.json'
    # print(f'======================= build nia 2022 summary. ========================')
    # summary_nia2022(src_path, csv_path, out_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # from_path = '/home2/channelbiome/NIA_Data/2022/unzip_221204'
    # to_path = '/home2/channelbiome/NIA_Data/2022/unzip_221130'
    # print(f'======================= start move folders and files. from = {from_path}, to = {to_path} ========================')
    # move_all(from_path, to_path, job_count=32)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/dev_gdrive/NIA_2022/Image_221130'
    # tar_path = '/home2/channelbiome/NIA_Data/2022/unzip_221130'
    # print(f'======================= start unzip and rename. src = {src_path}, tar = {tar_path} ========================')
    # unzip_rename_all(src_path, tar_path, job_count=32)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # yaml_file = '/home2/channelbiome/DataHub/docs/images_zip_info.yaml'
    # tar_path = '/home2/channelbiome/dev_gdrive/images'
    # out_file =  '/home2/channelbiome/DataHub/docs/lost_zip_info.yaml'
    # diff_zip_files(yaml_file, tar_path, out_file)

    # startTime = time.time()
    # src_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Image'
    # out_file = '/home2/channelbiome/NIA_Data/2021/blank_folder_info.yaml'
    # print(f'start check blank folder. path = {src_path}')
    # check_blank_folder(src_path, out_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Image'
    # in_file = '/home2/channelbiome/DataHub/docs/summary.json'
    # out_file = '/home2/channelbiome/NIA_Data/2021/file_disagree_info.json'
    # print(f'================ start check file count. path = {src_path} =====================')
    # check_file_count(src_path, in_file, out_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # image_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Image'
    # json_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Json'
    # in_file = '/home2/channelbiome/DataHub/docs/summary.json'
    # out_file = '/home2/channelbiome/NIA_Data/2021/image_json_compare.json'
    # print(f'================ start compare image and json. json = {json_path}\n'
    #       f'image = {image_path} =====================')
    # compare_image_json_folder(json_path, image_path, in_file, out_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/dev_gdrive/images'
    # tar_path = '/home2/channelbiome/NIA_Data/2021/temp_zip'
    # json_file = '/home2/channelbiome/NIA_Data/2021/image_json_compare.json'
    # # json_file = '/home2/channelbiome/NIA_Data/2021/file_disagree_info.json'
    # print(f'================ start unzip temp folder. src = {src_path}\n'
    #       f'tar = {tar_path} =====================')
    # upzip_temp_folder(src_path, tar_path, json_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/NIA_Data/2021/temp_zip'
    # tar_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Image'
    # json_file = '/home2/channelbiome/NIA_Data/2021/image_json_compare.json'
    # # json_file = '/home2/channelbiome/NIA_Data/2021/file_disagree_info.json'
    # print(f'================ start move from broken folder to right folder. src = {src_path}\n'
    #       f'tar = {tar_path} =====================')
    # move_broken_folder(src_path, tar_path, json_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/dev_gdrive/images'
    # tar_path = '/home2/channelbiome/NIA_Data/2021/temp_zip'
    # # tar_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Image'
    # json_file = '/home2/channelbiome/NIA_Data/2021/image_json_compare.json'
    # print(f'================ start unzip skipped file. src = {src_path}\n'
    #       f'tar = {tar_path} =====================')
    # unzip_skipped(src_path, tar_path, json_file)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/NIA_Data/2021/temp_zip'
    # tar_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Image'
    # print(f'================ start move and rename. src = {src_path}\n'
    #       f'tar = {tar_path} =====================')
    # move_and_rename(src_path, tar_path)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # zip_list = ['고구마줄기_221204.zip']
    # src_path = '/home2/channelbiome/NIA_Data/2022'
    # tar_path = '/home2/channelbiome/NIA_Data/2022/aaa'
    # unzip_rename(zip_list, src_path, tar_path)