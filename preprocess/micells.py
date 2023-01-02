import os
import sys
import time
from datetime import timedelta
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from functools import partial
import csv
import pandas as pd
import parmap
import json
import yaml
import numpy as np
import zipfile

def zip_folders(folder_list, tar_path):
    command = f'zip -q -r {os.path.join(tar_path, str(folder_list["index"]))}.zip'
    for folder in folder_list['list']:
        command += ' ' + f'"{folder}"'  # for the reason of 'xx json'
    os.system(command)


def zip_all(src_path, tar_path, zip_count=1):
    if not os.path.exists(tar_path):
        os.mkdir(tar_path)

    if zip_count == 1:
        os.chdir(src_path)
        os.system(f'zip -r {os.path.join(tar_path, os.path.split(src_path)[1])}.zip .')
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
        parmap.map(zip_folders, batches, tar_path=tar_path, pm_pbar=True, pm_processes=job_number)


def unzip_afile(filelist, src_path, tar_path, spread_folder, quite=False):
    pid = multiprocessing.current_process()
    for afile in tqdm(filelist, desc=f'upzip folders, pid = {pid.name}'):
        filename = os.path.splitext(afile)[0]
        command = f'unzip -O cp949'
        if quite:
            command += ' -qq'
        if spread_folder:
            command += f' {src_path}/{afile} -d {tar_path}'
        else:
            command += f' {src_path}/{afile} -d {tar_path}/{filename}'
        os.system(command)

# def unzip_afile_zipfile(filelist, src_path, tar_path, spread_folder, quite=False):
#     pip = multiprocessing.current_process()
#     for afile in tqdm(filelist, desc=f'unzip folder with zipfile lib, pid = {pip.name}'):
#         zip = f'{os.path.join(src_path, afile)}'
#         with zipfile.ZipFile(zip, 'r') as z:
#             path = tar_path if spread_folder else os.path.join(tar_path, afile)
#             z.extractall(path)


def unzip_afile_zipfile(filelist, src_path, tar_path, spread_folder, quite=False):
    pip = multiprocessing.current_process()
    for afile in tqdm(filelist, desc=f'unzip folder with zipfile lib, pid = {pip.name}'):
        zip = f'{os.path.join(src_path, afile)}'
        with zipfile.ZipFile(zip, 'r') as z:
            zipinfo = z.infolist()
            path = os.path.join(tar_path, afile.replace('.zip', ''))
            for info in zipinfo:
                info.filename = info.filename.encode('cp437').decode('euc-kr')
                z.extract(info, path)


def unzip_all(src_path, tar_path, remove_suffix=False, spread_folder=False, multi_process=False, quite=False,
              cpu_count=1):
    if remove_suffix:
        file_list = os.listdir(src_path)
        for afile in tqdm(file_list, desc='rename folders'):
            zipfile = afile.split(' ')[0]
            os.system(f'mv "{src_path}/{afile}" {src_path}/{zipfile}')

    if multi_process:
        filelist = os.listdir(src_path)
        job_number = cpu_count if cpu_count > 1 else multiprocessing.cpu_count()
        # job_number = 1
        total = len(filelist)
        chunk_size = total // job_number
        batches = chunks_list(filelist, chunk_size)
        pool = Pool(job_number)
        copy_func = partial(unzip_afile_zipfile, src_path=src_path, tar_path=tar_path, spread_folder=spread_folder, quite=quite)
        # copy_func = partial(unzip_afile, src_path=src_path, tar_path=tar_path, spread_folder=spread_folder, quite=quite)
        pool.map(copy_func, batches)
        print('Pool Ready')
        pool.close()
        pool.join()
        print('Pool Done')
    else:
        file_list = os.listdir(src_path)
        for afile in tqdm(file_list, desc='upzip folders'):
            filename = os.path.splitext(afile)[0]
            if spread_folder:
                os.system(f'unzip -O cp949 -qq -n {src_path}/{afile} -d {tar_path}')
            else:
                os.system(f'unzip -O cp949 -qq -n {src_path}/{afile} -d {tar_path}/{filename}')


def chunks_list(l, n):
    return [l[i:i + n] for i in range(0, len(l), n)]


def count_folders(root, save_dirname=False, windows=False):
    set_dir_top = set_dir_bottom = set()
    # set_dir_bottom = set()
    file_count = 0
    for r, d, f in os.walk(root):
        if r == root:
            set_dir_top.update(d)
        # elif '라벨링데이터' in r:
        #     print(f'라벨링데이터 상위 폴더 = {r}')
        else:
            set_dir_bottom.update(d)
        file_count += len(f)
    print(f'top dir = {len(set_dir_top)}, bottom dir = {len(set_dir_bottom)}, file = {file_count}')

    list_top = sorted(set_dir_top)
    list_bottom = sorted(set_dir_bottom)
    print(f'top dir names = {list_top}')
    print(f'bottom dir names = {list_bottom}')
    if save_dirname:
        encoding = 'utf-8-sig' if windows else 'utf8'
        dfTop = pd.DataFrame(list_top)
        dfTop.to_csv(os.path.join('../docs', 'top_dir_name.csv'), encoding=encoding, index=False, header=False)

        dfBottom = pd.DataFrame(list_bottom)
        dfBottom.to_csv(os.path.join('../docs', 'bottom_dir_name.csv'), encoding=encoding, index=False, header=False)


def move_files(srcpath, tarpath, dir_mv=False):
    if dir_mv:
        dir_list = list()
        for r, d, f in tqdm(list(os.walk(srcpath)), desc=f'iterating {srcpath}'):
            if r == srcpath:
                dir_list.extend(d)
            elif os.path.split(r)[1] in dir_list:
                tardir = os.path.split(r)[1]
                for dir in tqdm(d, f'move {r} {os.path.join(tarpath, tardir)}'):
                    com1 = f'rsync -a "{os.path.join(r, dir)}" "{tarpath}"'
                    os.system(com1)
                    com2 = f'rm -rf "{os.path.join(r, dir)}"'
                    os.system(com2)
    else:
        for r, d, f in tqdm(list(os.walk(srcpath)), desc=f'iterating {srcpath}'):
            if len(d) == 0:
                tardir = os.path.split(r)[1]
                for filename in tqdm(f, desc=f'move {r} {os.path.join(tarpath, tardir)}'):
                    command = f'mv "{os.path.join(r, filename)}" "{os.path.join(tarpath, tardir)}"'
                    os.system(command)

# dummy_header=['분류',    '대분류',    '중분류',  '소분류',   '기존코드','신규코드','음식명',     '코드',    '중복확인', '비고' '영양성분분석코드','분석명',  '학습폴더명',  '1']
dummy_header = ['cat', 'l_cat', 'm_cat', 's_cat', 'legacy', 'new', 'food_name', 'meta', 'dup',
                'etc','ingre_code', 'anal_name', 'train_name', 'dum4']
meta_header = ['food_name', 'ingre_code', 'meta', 'l_cat', 'm_cat', 's_cat', 'train_name']


def build_meta_table(file_path, as_name):
    df = pd.read_csv(file_path, names=dummy_header, usecols=meta_header, header=0)
    # df = df[~df.ingre_code.isnull()]
    df.to_csv(as_name, columns=meta_header, encoding='utf-8-sig', index=False)


def test_metal_table(file_path, food_name, ret_type='meta'):
    df = pd.read_csv(file_path, names=meta_header, header=0)
    if ret_type == 'meta':
        table = dict(zip(df.food_name, df.meta))
    elif ret_type == 'ingre_code':
        table = dict(zip(df.food_name, df.ingre_code))

    id = table.get(food_name)
    if id != None:
        return id
    else:
        return None


def load_meta_from_json(jsons_path, table_path, nutrients_names, meta_header):
    df = pd.read_csv(table_path, names=meta_header, header=0)
    dic_replace_name = dict(zip(df.food_name, df.train_name))
    with open(nutrients_names, 'r') as f:
        nutrients_json = json.load(f)

    nutrients_list = list(nutrients_json.keys())

    json_list = os.listdir(jsons_path)
    dic_meta = {}
    for jsonfile in json_list:
        with open(os.path.join(jsons_path, jsonfile), 'r') as f:
            json_info = json.load(f)

        class_name = json_info['Name']
        new_name = dic_replace_name.get(class_name)
        if new_name is not None:
            class_name = new_name

        ingre_info = [json_info[ingre] for ingre in nutrients_list]
        dic_meta[class_name] = ingre_info

    return dic_meta, nutrients_list


jsonTemplate = {"Code Name": "xxx",
                "Name": "xxx",
                "W": "xxx",
                "H": "xxx",
                "File Format": "xxx",
                "Cat 1": "01",
                "Cat 2": "02",
                "Cat 3": "03",
                "Point(x,y)": "xxx",
                "Camera Angle": "T",
                "Meta File": "xx"}


# meta_header = ['food_name', 'ingre_code', 'meta', 'l_cat', 'm_cat', 's_cat', 'train_name']
def cvrt_json_all(root_path, meta_path, yaml_path, table_check=False):
    df = pd.read_csv(meta_path, names=meta_header, header=0)
    df.food_name = np.where(df.train_name.isnull(), df.food_name, df.train_name)
    meta_table = dict(zip(df.food_name, df.meta))
    l_table = dict(zip(df.food_name, df.l_cat))
    m_table = dict(zip(df.food_name, df.m_cat))
    s_table = dict(zip(df.food_name, df.s_cat))

    with open(yaml_path, 'r') as f:
        yalm_data = yaml.load(f, Loader=yaml.FullLoader)

    class_list = yalm_data['names']

    if table_check:
        test_good = True
        for class_name in class_list:
            if meta_table.get(class_name) == None:
                print(f'There is no key {class_name}')
                test_good = False
        if test_good == False:
            return

    save_path = os.path.join(root_path, 'jsons_new')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    label_path = os.path.join(root_path, 'labels')
    txt_list = os.listdir(label_path)
    for label_file in txt_list:
        jsonlist = []
        with open(os.path.join(label_path, label_file), 'r') as pf:
            lines = pf.readlines()
            file_name = os.path.split(label_file)[1]
            for aline in lines:
                try:
                    contents = aline.strip().split(' ')
                    class_name = class_list[int(contents[0])]
                    jsonCopy = jsonTemplate.copy()
                    jsonCopy["Code Name"] = file_name.replace('.txt', '.jpg')
                    jsonCopy["Name"] = class_name
                    jsonCopy["Point(x,y)"] = f'{contents[1]},{contents[2]}'
                    jsonCopy["W"] = contents[3]
                    jsonCopy["H"] = contents[4]
                    jsonCopy["File Format"] = 'jpg'  # 나중에 할 때에는 상황에 맞게 처리 한다. 지금 처리 해도 어짜피 고쳐야 한다.
                    jsonCopy["Cat 1"] = l_table[class_name]
                    jsonCopy["Cat 2"] = m_table[class_name]
                    jsonCopy["Cat 3"] = f'{s_table[class_name]:03}'
                    jsonCopy["Meta File"] = f'{meta_table[class_name]}_meta.json'
                    jsonlist.append(jsonCopy)
                except Exception as e:
                    print(f'Exception = {e}')

            with open(os.path.join(save_path, file_name.replace('.txt', '.json')), 'w', encoding='utf-8') as f:
                json.dump(jsonlist, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    startTime = time.time()
    # src_path = '/home2/channelbiome/dev_gdrive/unzip_test'
    # src_path = '/home2/channelbiome/dev_gdrive/NIA_2022'
    # tar_path = '/home2/channelbiome/NIA_Data/2022/221219'
    src_path = "/home2/channelbiome/NIA_Zips/test_unzip"
    tar_path = "/home2/channelbiome/NIA_Zips/test_unzip_target"
    print(f'Start unzip folder = {src_path}')
    unzip_all(src_path, tar_path, remove_suffix=False, spread_folder=False, multi_process=True, quite=True, cpu_count=2)
    # unzip_all(src_path, tar_path, remove_suffix=False, spread_folder=True, multi_process=True, quite=True)
    endTime = time.time()
    print(f'Elapsed Time = {timedelta(seconds=endTime-startTime)}')

    # / home2 / channelbiome / food_image / NIA_2021_Json_SF
    # startTime = time.time()
    # tar_path = sys.argv[1]
    # count_folders(tar_path, save_dirname=True, windows=True)
    # endTime = time.time()

    # startTime = time.time()
    # src_path = sys.argv[1]
    # tar_path = sys.argv[2]
    # move_files(src_path, tar_path, dir_mv=False)
    # endTime = time.time()

    # /home2/channelbiome/food_image/NIA_2021_Relabeled /home2/channelbiome/food_image/NIA_2021_Relabeled_Zip
    # /home2/channelbiome/DataHub/NIA_20221014_processed/train /home2/channelbiome/food_image/NIA_2021_Relabeled_Zip
    # startTime = time.time()
    # src_path = '/home2/channelbiome/DataHub/NIA_20221014_processed/train'
    # tar_path = '/home2/channelbiome/NIA_Zips/NIA_221014_cvrt_json'
    # # zip_all(src_path, tar_path, zip_count=1)
    # zip_all(src_path, tar_path, zip_count=10)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    path = './web/data/FoodRawTable.csv'
    as_name = './web/data/FoodMetaTable.csv'
    build_meta_table(path, as_name)

    # startTime = time.time()
    # # root_path = '/home2/channelbiome/DataHub/NIA_20221014_processed/test'
    # # root_path = '/home2/channelbiome/DataHub/NIA_20221014_processed/valid'
    # root_path = '/home2/channelbiome/DataHub/NIA_20221014_processed/train'
    # # root_path = '/home2/channelbiome/DataHub/NIA_20221014_processed/cvrt_test'
    # meta_path = './web/data/FoodMetaTable.csv'
    # yaml_path = '/home2/channelbiome/DataHub/NIA_20221014_processed/data.yaml'
    # print(f'===============start cvrt json {root_path} ====================')
    # cvrt_json_all(root_path, meta_path, yaml_path, table_check=True)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # table_path = './web/data/FoodMetaTable.csv'
    # jsons_path = './web/data/meta_data'
    # ingre_names = './web/data/ingredients_meta.json'
    # meta_table, ingre_list = load_meta_from_json(jsons_path, table_path, ingre_names, meta_header)
    # meta_info = meta_table.get('마늘빵')
    # if meta_info == None:
    #     pass
    # else:
    #     print(dict(zip(ingre_list, meta_info)))
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

