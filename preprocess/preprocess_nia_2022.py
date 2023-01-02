import json
import sys
import os
import time
import datetime
import shutil
from datetime import timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import micells
from itertools import repeat
import parmap
from PIL import Image

json_template = {
    "Code Name": "xxx",
    "Name": "xxx",
    "W": "0.0",
    "H": "0.0",
    "File Format": "xxx",
    "Cat 1": "xxx",
    "Cat 2": "xxx",
    "Cat 3": "xxx",
    "Point(x,y)": "0.0,0.0",
    "Camera Angle": "T",
    "Meta File": "xxx_meta.json",
    "Source": "NIA 2021"
}

IMAGE_SOURCE = 'NIA 2021'


def build_nia2022json(dir_list, src_path, meta_dict):
    error_list = []
    for dir in dir_list:
        class_name = dir
        file_list = os.listdir(os.path.join(src_path, dir))
        txt_list = [file for file in file_list if os.path.splitext(file)[1] == '.txt']
        for txt_file in txt_list:
            with open(os.path.join(src_path, dir, txt_file), 'r') as pf:
                lines = pf.readlines()

            json_list = []
            for aline in lines:
                contents = aline.strip().split(' ')
                try:
                    json_copy = json_template.copy()
                    json_copy['Code Name'] = f'{txt_file.replace("txt", "jpg")}'
                    json_copy['Name'] = class_name
                    json_copy['W'] = contents[3]
                    json_copy['H'] = contents[4]
                    json_copy['File Format'] = 'jpg'
                    json_copy['Cat 1'] = meta_dict[class_name][1]
                    json_copy['Cat 2'] = meta_dict[class_name][2]
                    json_copy['Cat 3'] = meta_dict[class_name][3]
                    json_copy['Point(x,y)'] = f'{contents[1]},{contents[2]}'
                    json_copy['Camera Angle'] = 'T'
                    json_copy['Meta File'] = f'{meta_dict[class_name][0]}_meta.json'
                    json_copy['Source'] = 'xxx'
                    json_list.append(json_copy)
                except Exception as e:
                    msg = f'Exception, file = f{os.path.join(src_path, dir, txt_file)}, reason = {e}'
                    error_list.append(msg + '\n')
                    continue

            with open(os.path.join(src_path, dir, f'{txt_file.replace("txt", "json")}'), 'w', encoding='utf-8') as f:
                json.dump(json_list, f, indent=2, ensure_ascii=False)


def cvrt_nia2021json(dir_list, src_path, tar_path, meta_dict):
    pid = multiprocessing.current_process()
    error_list = []
    for dir in dir_list:
        class_name = dir.replace(' json', '')
        file_list = os.listdir(os.path.join(src_path, dir))
        for file in file_list:
            with open(os.path.join(src_path, dir, file), 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            json_list = []
            for json_src in json_data:
                try:
                    json_copy = json_template.copy()
                    # if json_src['Name'] != class_name
                    # class_name = json_src['Name']
                    json_copy['Code Name'] = json_src['Code Name']
                    json_copy['Name'] = class_name  # json_src['Name'] is not always correct. class_name is better.
                    json_copy['W'] = json_src['W']
                    json_copy['H'] = json_src['H']
                    json_copy['File Format'] = json_src['File Format']
                    json_copy['Cat 1'] = meta_dict[class_name][1]
                    json_copy['Cat 2'] = meta_dict[class_name][2]
                    json_copy['Cat 3'] = meta_dict[class_name][3]
                    json_copy['Point(x,y)'] = json_src['Point(x,y)']
                    json_copy['Camera Angle'] = 'T'
                    json_copy['Meta File'] = f'{meta_dict[class_name][0]}_meta.json'
                    json_copy['Source'] = IMAGE_SOURCE
                    json_list.append(json_copy)
                except Exception as e:
                    msg = f'Exception, file = f{os.path.join(src_path, dir, file)}, reason = {e}'
                    error_list.append(msg + '\n')
                    continue

            if not os.path.exists(os.path.join(tar_path, dir)):
                os.mkdir(os.path.join(tar_path, dir))

            with open(os.path.join(tar_path, dir, file), 'w', encoding='utf-8') as f:
                json.dump(json_list, f, indent=2, ensure_ascii=False)

    if len(error_list) != 0:
        with open(os.path.join(tar_path, f'error_{pid.name}.txt'), 'w') as f:
            f.writelines(error_list)


def drop_mismatch(img_dirs, img_path, json_path, drop_path):
    drop_list = []
    for img_dir in img_dirs:
        img_files = [os.path.splitext(img)[0] for img in os.listdir(os.path.join(img_path, img_dir))]
        json_files = [os.path.splitext(ajson)[0] for ajson in os.listdir(os.path.join(json_path, f'{img_dir} json'))]
        img_only = list(set(img_files) - set(json_files))
        json_only = list(set(json_files) - set(img_files))
        if len(img_only) != 0:
            drop_img_path = os.path.join(drop_path, img_dir)
            if not os.path.exists(drop_img_path):
                os.mkdir(drop_img_path)
            img_only_ext = [f'{img}.jpg' for img in img_only]
            for img in img_only_ext:
                drop_list.append(os.path.join(drop_img_path, img))
                shutil.move(os.path.join(img_path, img_dir, img), os.path.join(drop_img_path))

        if len(json_only) != 0:
            from_json_path = os.path.join(json_path, f'{img_dir} json')
            drop_json_path = os.path.join(drop_path, f'{img_dir} json')
            if not os.path.exists(drop_json_path):
                os.mkdir(drop_json_path)
            json_only_ext = [f'{ajson}.json' for ajson in json_only]
            for ajson in json_only_ext:
                drop_list.append(os.path.join(drop_json_path, ajson))
                shutil.move(os.path.join(from_json_path, ajson), os.path.join(drop_json_path))

    return drop_list


def check_corrupt(folder_list, src_path, display_error, label_check):
    corrupt_list = []
    malformat_list = []
    wronglabel_list = []
    for folder in folder_list:
        img_files = [file for file in os.listdir(os.path.join(src_path, folder)) if os.path.splitext(file)[1] != '.txt'
                     and os.path.splitext(file)[1] != '.json']
        for img in img_files:
            if os.path.splitext(img)[1] != '.jpg':
                if display_error:
                    print(f'format error = {os.path.join(folder, img)}')
                malformat_list.append(os.path.join(folder, img))
            try:
                _img = Image.open(os.path.join(src_path, folder, img))
                _img.verify()
                _img.close()
            except Exception as e:
                if display_error:
                    print(f'file corrupt = {e}')
                corrupt_list.append(os.path.join(folder, img))

        if label_check:
            txt_files = [file for file in os.listdir(os.path.join(src_path, folder)) if
                         os.path.splitext(file)[1] == '.txt']
            for txt in txt_files:
                with open(os.path.join(src_path, folder, txt), 'r') as pf:
                    lines = pf.readlines()

                for aline in lines:
                    contents = [float(x) for x in aline.split(' ')[1:]]
                    wrong_label = any(value < 0 or value > 1 for value in contents)
                    if wrong_label is True:
                        if display_error:
                            print(f'wrong label = {os.path.join(folder, txt)}')
                        wronglabel_list.append(os.path.join(folder, txt))
                        break

    return {'corrupt files': corrupt_list, 'malformat files': malformat_list, 'wrong label files': wronglabel_list}


def copyfiles_nia2022(file_list, tar_img, tar_json, tar_txt):
    for file in file_list:
        try:
            src_img, src_json, src_txt = f'{file}.jpg', f'{file}.json', f'{file}.txt'
            shutil.copy(src_img, tar_img)
            # shutil.copy(src_json, tar_json)
            shutil.copy(src_txt, tar_txt)
        except Exception as e:
            print(f'copyfiles_nia2022 exception = {e}')


def copyfiles_nia2021(file_list, img_path, json_path, txt_path, tar_img, tar_json, tar_txt):
    for file in file_list:
        try:
            mid, last = os.path.split(file)
            src_img = os.path.join(img_path, f'{mid}', f'{last}.jpg')
            src_json = os.path.join(json_path, f'{mid} json', f'{last}.json')
            src_txt = os.path.join(txt_path, f'{mid} txt', f'{last}.txt')
            shutil.copy(src_img, tar_img)
            # shutil.copy(src_json, tar_json)
            shutil.copy(src_txt, tar_txt)
        except Exception as e:
            print(f'copyfiles_nia2021 exception = {e}')


def yolo_update(dir_list, src_path, class_names):
    pid = multiprocessing.current_process()
    for dir in tqdm(dir_list, desc=f'yolo update {pid.name}'):
        class_name = dir
        file_list = os.listdir(os.path.join(src_path, dir))
        file_list = [file for file in file_list if os.path.splitext(file)[1] == '.txt']
        for file in file_list:
            txt_path = os.path.join(src_path, dir, file)
            try:
                with open(txt_path, 'r', encoding='utf-8') as pf:
                    lines = pf.readlines()

                new_lines = []
                for aline in lines:
                    contents = aline.split(' ')
                    contents[0] = str(class_names.index(class_name))
                    new_lines.append(' '.join(contents))

                with open(txt_path, 'w') as pf:
                    pf.writelines(new_lines)
            except Exception as e:
                print(f'yolo update error, file = {txt_path}, reason = {e}')


def yolo_create(dir_list, json_path, txt_path, class_names):
    pid = multiprocessing.current_process()
    for dir in tqdm(dir_list, desc=f'yolo create {pid.name}'):
        class_name = dir.replace(' json', '')
        file_list = os.listdir(os.path.join(json_path, dir))
        txt_dir = os.path.join(txt_path, f'{class_name} txt')
        if not os.path.exists(txt_dir): os.mkdir(txt_dir)
        for file in file_list:
            json_file = os.path.join(json_path, dir, file)
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)

                lines = []
                for ajson in json_data:
                    center = ajson['Point(x,y)']
                    center = center.split(',')
                    W = ajson['W']
                    H = ajson['H']
                    lines.append(f'{class_names.index(class_name)} {center[0]} {center[1]} {W} {H}')

                txt_file = os.path.join(txt_dir, f'{os.path.splitext(file)[0]}.txt')
                with open(txt_file, 'w') as pf:
                    pf.writelines(lines)
            except Exception as e:
                print(f'yolo create error, file = {json_path}, reason = {e}')


def rename_file(dict_info, src_path):
    pid = multiprocessing.current_process()
    img_path = os.path.join(src_path, 'images')
    json_path = os.path.join(src_path, 'jsons')
    label_path = os.path.join(src_path, 'labels')
    log_list = []
    img_list = dict_info['list']
    for index, img in enumerate(tqdm(img_list, desc=f'rename file pid = {pid.name}')):
        # with open(os.path.join(json_path, img.replace(".jpg", ".json")), 'r', encoding='utf-8') as f:
        #     json_data = json.load(f)
        # json_data = json_data[0]
        file_name = f"{dict_info['index']+1:04d}_{index+1:06d}"
        os.rename(os.path.join(img_path, img), os.path.join(img_path, f'{file_name}.jpg'))
        os.rename(os.path.join(json_path, img.replace(".jpg", ".json")),
                  os.path.join(json_path, f'{file_name}.json'))
        os.rename(os.path.join(label_path, img.replace(".jpg", ".txt")),
                  os.path.join(label_path, f'{file_name}.txt'))
        log_list.append((img.replace('.jpg', ''), file_name))

    return log_list

class Nia2022DataBulder:
    def __init__(self, basedir):
        self.basedir = basedir
        self.errlog = open(os.path.join(self.basedir, 'errorlog.txt'), 'w')

    def __del__(self):
        self.errlog.close()

    def check_json_classname_from_meta(self, src_path, meta_path, out_file):
        df = pd.read_csv(meta_path, names=micells.meta_header, header=0)
        meta_dict = dict(zip(df.food_name, df.meta))

        noname_list = []
        dir_list = os.listdir(src_path)
        for dir in tqdm(dir_list, desc=f'Check Json"s class name from meta table'):
            class_name = dir.replace(' json', '')
            info = meta_dict.get(class_name)
            if info is None:
                noname_list.append(class_name)
                print(f'noname, class name is {class_name}')

        df = pd.DataFrame(noname_list, columns=['no name class'])
        df.to_csv(out_file, index=False, encoding='utf-8')

    def rename_folder_name(self, src_path, rep_table, folder_mode='image'):
        header = ['origin', 'rep']
        df = pd.read_csv(rep_table, names=header, header=0, encoding='utf-8')
        rep_dict = dict(zip(df.origin, df.rep))

        dir_list = os.listdir(src_path)
        for dir in tqdm(dir_list, desc=f'Check Json"s class name from meta table'):
            class_name = dir.replace(' json', '')
            origin_name = rep_dict.get(class_name)
            if origin_name is not None:
                tar_name = origin_name if folder_mode == 'image' else f'{origin_name} json'
                os.rename(f'{os.path.join(src_path, dir)}', f'{os.path.join(src_path, tar_name)}')

    def drop_mismatching_nia2021(self, img_path, json_path, drop_path, out_file, job_count=1):
        img_dirs = os.listdir(img_path)
        total = len(img_dirs)
        chunk_size = total // job_count
        batches = micells.chunks_list(img_dirs, chunk_size)
        drop_lists = parmap.map(drop_mismatch, batches, img_path=img_path, json_path=json_path, drop_path=drop_path,
                                pm_pbar=True, pm_processes=job_count)

        print(f'drop mismatch data finished')

        drop_list = []
        for alist in drop_lists:
            drop_list = drop_list + alist

        if len(drop_list) != 0:
            drop_list.sort()
            df = pd.DataFrame(drop_list, columns=['mismatch files'])
            df.to_csv(out_file, index=False, encoding='utf-8')

    def cvrt_nia2021json(self, src_path, tar_path, meta_path, job_count=1):
        df = pd.read_csv(meta_path, names=micells.meta_header, header=0)
        meta_dict = dict(zip(df.food_name, zip(df.meta, df.l_cat, df.m_cat, df.s_cat)))

        dir_list = os.listdir(src_path)
        total = len(dir_list)
        chunk_size = total // job_count
        batches = micells.chunks_list(dir_list, chunk_size)
        parmap.map(cvrt_nia2021json, batches, src_path=src_path, tar_path=tar_path, meta_dict=meta_dict, pm_pbar=True,
                   pm_processes=job_count)

    def build_nia2022json(self, src_path, meta_path, job_count=1):
        df = pd.read_csv(meta_path, names=micells.meta_header, header=0)
        meta_dict = dict(zip(df.food_name, zip(df.meta, df.l_cat, df.m_cat, df.s_cat)))

        dir_list = os.listdir(src_path)
        total = len(dir_list)
        chunk_size = total // job_count
        batches = micells.chunks_list(dir_list, chunk_size)
        parmap.map(build_nia2022json, batches, src_path=src_path, meta_dict=meta_dict, pm_pbar=True,
                   pm_processes=job_count)

    # if json_path is None this will be nia 2022 data.
    def build_raw_spec_file(self, yaml_path, img_path, json_path=None):
        is_nia_2022 = True if json_path is None else False
        yaml_out = {}
        class_spec = {}
        tot_img_count, tot_json_count = 0, 0
        img_dirs = os.listdir(img_path)
        img_dirs.sort()
        for img_dir in tqdm(img_dirs, desc=f'build raw spec file'):
            if is_nia_2022:
                file_list = os.listdir(os.path.join(img_path, img_dir))
                img_list = [file for file in file_list if os.path.splitext(file)[1] == '.jpg']
                json_list = [file for file in file_list if os.path.splitext(file)[1] == '.json']
            else:
                img_list = os.listdir(os.path.join(img_path, img_dir))
                json_list = os.listdir(os.path.join(json_path, f'{img_dir} json'))
            tot_img_count += len(img_list)
            tot_json_count += len(json_list)
            class_spec[img_dir] = len(img_list)

        class_spec = {k: v for k, v in sorted(class_spec.items(), key=lambda item: item[1])}
        yaml_out['source information'] = {'image directory': img_path, 'json directory': json_path}
        yaml_out['class count'] = len(img_dirs)
        yaml_out['total image count'] = tot_img_count
        yaml_out['total json count'] = tot_json_count
        yaml_out['class spec'] = class_spec
        yaml_out['directories'] = img_dirs

        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    '''
    Referring to train_spec.yaml, build data.yaml and train/valid/test data folder.
    # 1. nia2022 update yolo txt files and nia2021 create yolo txt files 
    # 2. train/val/test split
    # 3. copy files from images/json folder to image/json of train/val/test folder
    # 4. create data.yaml and detail_spec.yaml file.
    '''

    def build_dataset_from_spec(self, spec_file, save_split, job_count):
        with open(spec_file, 'r') as f:
            yalm_data = yaml.load(f, Loader=yaml.FullLoader)

        nia_2022 = yalm_data.get('nia 2022')
        nia_2021 = yalm_data.get('nia 2021')
        if nia_2021 is None and nia_2022 is None:
            print(f'There is no information in {spec_file}')
            return

        class_names = nia_2022['names'] if nia_2022 is not None else []
        if nia_2021 is not None: class_names += nia_2021['names']

        # 1. nia2022 update yolo txt files and nia2021 create yolo txt files
        # 1.1 nia 2022 update txt file
        if nia_2022 is not None:
            # img_path = nia_2022['image directory']
            # img_list = os.listdir(img_path)
            # for img_dir in tqdm(img_list, desc=f'nia 2022 txt file update'):
            #     class_name = img_dir
            #     file_list = os.listdir(os.path.join(img_path, img_dir))
            #     txt_list = [file for file in file_list if os.path.splitext(file)[1] == '.txt']
            #     for txt_file in txt_list:
            #         txt_path = os.path.join(img_path, img_dir, txt_file)
            #         self._yolo_update(txt_path, class_name, class_names)
            img_path = nia_2022['image directory']
            dir_list = os.listdir(img_path)
            total = len(dir_list)
            chunk_size = total // job_count
            batches = micells.chunks_list(dir_list, chunk_size)
            parmap.map(yolo_update, batches, src_path=img_path, class_names=class_names,
                       pm_pbar={'desc': f'nia 2022 yolo file update'}, pm_processes=job_count)

        # 1.2 nia 2021 update txt file
        if nia_2021 is not None:
            # json_path = nia_2021['json directory']
            # txt_path = nia_2021['txt directory']
            # if not os.path.exists(txt_path): os.mkdir(txt_path)
            # json_list = os.listdir(json_path)
            # for json_dir in tqdm(json_list, desc=f'nia 2021 txt file create'):
            #     class_name = json_dir.replace(' json', '')
            #     json_files = os.listdir(os.path.join(json_path, json_dir))
            #     txt_dir = os.path.join(txt_path, f'{class_name} txt')
            #     if not os.path.exists(txt_dir): os.mkdir(txt_dir)
            #     for json_file in json_files:
            #         _json_path = os.path.join(json_path, json_dir, json_file)
            #         _txt_path = os.path.join(txt_dir, f'{os.path.splitext(json_file)[0]}.txt')
            #         self._yolo_create(_json_path, _txt_path, class_name, class_names)
            json_path = nia_2021['json directory']
            txt_path = nia_2021['txt directory']
            if not os.path.exists(txt_path): os.mkdir(txt_path)
            dir_list = os.listdir(json_path)
            total = len(dir_list)
            chunk_size = total // job_count
            batches = micells.chunks_list(dir_list, chunk_size)
            parmap.map(yolo_create, batches, json_path=json_path, txt_path=txt_path, class_names=class_names,
                       pm_pbar={'desc': f'nia 2021 yolo file create'}, pm_processes=job_count)



        # 2. train/val/test split
        dataset_spec, class_spec = {}, {}
        upper = sys.maxsize if yalm_data['train bounds']['upper'] is None else yalm_data['train bounds']['upper']
        # lower = yalm_data['train bounds']['lower']
        train_log, test_log, val_log = [], [], []
        # 2.1 nia 2022
        if nia_2022 is not None:
            img_path = nia_2022['image directory']
            img_list = os.listdir(img_path)
            train_nia2022, test_nia2022, val_nia2022 = [], [], []
            for img_dir in tqdm(img_list, desc=f'nia 2022 file path split'):
                file_list = os.listdir(os.path.join(img_path, img_dir))
                # image split
                img_list = [os.path.join(img_path, img_dir, os.path.splitext(file)[0])
                            for index, file in enumerate(file_list) if
                            os.path.splitext(file)[1] == '.jpg' and index < upper]

                train_files, test_files = train_test_split(img_list, train_size=yalm_data['data_ratio']['train'],
                                                           random_state=42, shuffle=True)
                ratio = yalm_data['data_ratio']['val'] / (
                        yalm_data['data_ratio']['test'] + yalm_data['data_ratio']['val'])
                val_files, test_files = train_test_split(test_files, test_size=ratio, random_state=42, shuffle=True)
                train_nia2022 += train_files
                test_nia2022 += test_files
                val_nia2022 += val_files
                class_spec[img_dir] = {'total': len(img_list), 'train': len(train_files),
                                       'valid': len(val_files), 'test': len(test_files)}
            if save_split:
                train_log = train_log + [file for file in train_nia2022]
                test_log = test_log + [file for file in test_nia2022]
                val_log = val_log + [file for file in val_nia2022]

        # 2.2 nia 2021
        if nia_2021 is not None:
            train_nia2021, test_nia2021, val_nia2021 = [], [], []
            img_path = nia_2021['image directory']
            img_list = os.listdir(img_path)
            for img_dir in tqdm(img_list, desc=f'nia 2021 file path split'):
                file_list = os.listdir(os.path.join(img_path, img_dir))
                # image split
                img_list = [os.path.join(img_dir, os.path.splitext(file)[0])
                            for index, file in enumerate(file_list) if index < upper]

                train_files, test_files = train_test_split(img_list, train_size=yalm_data['data_ratio']['train'],
                                                           random_state=42, shuffle=True)
                ratio = yalm_data['data_ratio']['val'] / (
                        yalm_data['data_ratio']['test'] + yalm_data['data_ratio']['val'])
                val_files, test_files = train_test_split(test_files, test_size=ratio, random_state=42, shuffle=True)
                train_nia2021 += train_files
                test_nia2021 += test_files
                val_nia2021 += val_files
                class_spec[img_dir] = {'total': len(img_list), 'train': len(train_files),
                                       'valid': len(val_files), 'test': len(test_files)}
            if save_split:
                train_log = train_log + [os.path.join(img_dir, os.path.split(file)[1]) for file in train_nia2021]
                test_log = test_log + [os.path.join(img_dir, os.path.split(file)[1]) for file in test_nia2021]
                val_log = val_log + [os.path.join(img_dir, os.path.split(file)[1]) for file in val_nia2021]

        if save_split:
            train_log.sort()
            df = pd.DataFrame(train_log, columns=['train files'])
            df.to_csv(os.path.join(self.basedir, "train_data.csv"), index=False, encoding='utf-8')

            test_log.sort()
            df = pd.DataFrame(test_log, columns=['test files'])
            df.to_csv(os.path.join(self.basedir, "test_data.csv"), index=False, encoding='utf-8')

            val_log.sort()
            df = pd.DataFrame(val_log, columns=['validation files'])
            df.to_csv(os.path.join(self.basedir, "valid_data.csv"), index=False, encoding='utf-8')

        print(f'split done!!')

        # 3. copy files from images/json folder to image/json of train/val/test folder
        basedir = self.basedir
        train_images = os.path.join(basedir, yalm_data['folder']['train'], 'images')
        train_labels = os.path.join(basedir, yalm_data['folder']['train'], 'labels')
        train_jsons = os.path.join(basedir, yalm_data['folder']['train'], 'jsons')
        val_images = os.path.join(basedir, yalm_data['folder']['val'], 'images')
        val_labels = os.path.join(basedir, yalm_data['folder']['val'], 'labels')
        val_jsons = os.path.join(basedir, yalm_data['folder']['val'], 'jsons')
        test_images = os.path.join(basedir, yalm_data['folder']['test'], 'images')
        test_labels = os.path.join(basedir, yalm_data['folder']['test'], 'labels')
        test_jsons = os.path.join(basedir, yalm_data['folder']['test'], 'jsons')

        if not os.path.exists(train_images): os.makedirs(train_images)
        if not os.path.exists(train_labels): os.makedirs(train_labels)
        if not os.path.exists(train_jsons): os.makedirs(train_jsons)
        if not os.path.exists(val_images): os.makedirs(val_images)
        if not os.path.exists(val_labels): os.makedirs(val_labels)
        if not os.path.exists(val_jsons): os.makedirs(val_jsons)
        if not os.path.exists(test_images): os.makedirs(test_images)
        if not os.path.exists(test_labels): os.makedirs(test_labels)
        if not os.path.exists(test_jsons): os.makedirs(test_jsons)

        # 3.1 copy nia 2022
        if nia_2022 is not None:
            self._copy_files_nia2022(test_nia2022, test_images, test_jsons, test_labels, 'nia 2022 test copy',
                                     job_count)
            self._copy_files_nia2022(val_nia2022, val_images, val_jsons, val_labels, 'nia 2022 valid copy', job_count)
            self._copy_files_nia2022(train_nia2022, train_images, train_jsons, train_labels, 'nia 2022 train copy',
                                     job_count)

    # 3.2 copy nia 2021
        if nia_2021 is not None:
            img_path = nia_2021['image directory']
            json_path = nia_2021['json directory']
            txt_path = nia_2021['txt directory']
            self._copy_files_nia2021(test_nia2021, img_path, json_path, txt_path,
                                     test_images, test_jsons, test_labels, 'nia 2021 test copy', job_count)
            self._copy_files_nia2021(val_nia2021, img_path, json_path, txt_path,
                                     val_images, val_jsons, val_labels, 'nia 2021 valid copy', job_count)

            self._copy_files_nia2021(train_nia2021, img_path, json_path, txt_path,
                                     train_images, train_jsons, train_labels, 'nia 2021 train copy', job_count)

        # 4. create data.yaml and detail_spec.yaml file.
        # 4.1 data.yaml
        yalm_out = dict()
        yalm_out['path'] = os.path.abspath(basedir)
        yalm_out['train'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['train']))
        yalm_out['val'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['val']))
        yalm_out['test'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['test']))
        yalm_out['nc'] = len(class_names)
        yalm_out['names'] = class_names

        with open(os.path.join(basedir, 'data.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(yalm_out, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        # 4.2 detail_spec.yaml
        class_spec = {k: v for k, v in sorted(class_spec.items(), key=lambda item: item[0])}
        if nia_2022 is not None:
            tr_2022, te_2022, val_2022 = len(train_nia2022), len(test_nia2022), len(val_nia2022)
        else:
            tr_2022, te_2022, val_2022 = 0, 0, 0
        if nia_2021 is not None:
            tr_2021, te_2021, val_2021 = len(train_nia2021), len(test_nia2021), len(val_nia2021)
        else:
            tr_2021, te_2021, val_2021 = 0, 0, 0

        dataset_spec['total image count'] = tr_2021 + te_2021 + val_2021 + tr_2022 + te_2022 + val_2022
        dataset_spec['total train count'] = tr_2021 + tr_2022
        dataset_spec['total valid count'] = val_2021 + val_2022
        dataset_spec['total test count'] = te_2021 + te_2022
        dataset_spec['class count'] = len(class_spec.keys())
        dataset_spec['class split info'] = class_spec

        with open(os.path.join(basedir, 'dataset_spec.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(dataset_spec, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _copy_files_nia2022(self, file_list, tar_img, tar_json, tar_txt, desc, job_count):
        total = len(file_list)
        chunk_size = total // job_count
        batches = micells.chunks_list(file_list, chunk_size)
        parmap.map(copyfiles_nia2022, batches,
                   tar_img=tar_img, tar_json=tar_json, tar_txt=tar_txt,
                   pm_pbar={'desc': desc}, pm_processes=job_count)

    def _copy_files_nia2021(self, file_list, img_path, json_path, txt_path, tar_img, tar_json, tar_txt, desc,
                            job_count):
        total = len(file_list)
        chunk_size = total // job_count
        batches = micells.chunks_list(file_list, chunk_size)
        parmap.map(copyfiles_nia2021, batches,
                   img_path=img_path, json_path=json_path, txt_path=txt_path,
                   tar_img=tar_img, tar_json=tar_json, tar_txt=tar_txt,
                   pm_pbar={'desc': desc}, pm_processes=job_count)

    def _yolo_create(self, json_path, txt_path, class_name, class_names):
        with open(json_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        lines = []
        for ajson in json_data:
            center = ajson['Point(x,y)']
            center = center.split(',')
            W = ajson['W']
            H = ajson['H']
            lines.append(f'{class_names.index(class_name)} {center[0]} {center[1]} {W} {H}')

        with open(txt_path, 'w') as pf:
            pf.writelines(lines)

    def _yolo_update(self, txt_path, class_name, class_names):
        with open(txt_path, 'r') as pf:
            lines = pf.readlines()

        new_lines = []
        for aline in lines:
            contents = aline.split(' ')
            contents[0] = str(class_names.index(class_name))
            new_lines.append(' '.join(contents))

        with open(txt_path, 'w') as pf:
            pf.writelines(new_lines)

    def check_image_corrupt(self, src_path, corrupt_file, malformat_file, wronglabel_file, job_count,
                            display_error=True, label_check=True):
        dir_list = os.listdir(src_path)
        total = len(dir_list)
        chunk_size = total // job_count
        batches = micells.chunks_list(dir_list, chunk_size)
        error_list = parmap.map(check_corrupt, batches, src_path=src_path,
                                display_error=display_error, label_check=label_check,
                                pm_pbar=True, pm_processes=job_count)

        print(f'check image corrupt finished')

        corrupt_list, malformat_list, wronglabel_list = [], [], []
        for adict in error_list:
            corrupt_list = corrupt_list + adict['corrupt files']
            malformat_list = malformat_list + adict['malformat files']
            wronglabel_list = wronglabel_list + adict['wrong label files']

        if len(corrupt_list) != 0:
            corrupt_list.sort()
            df = pd.DataFrame(corrupt_list, columns=['corrupt files'])
            df.to_csv(corrupt_file, index=False, encoding='utf-8')

        if len(malformat_list) != 0:
            malformat_list.sort()
            df = pd.DataFrame(malformat_list, columns=['malformat files'])
            df.to_csv(malformat_file, index=False, encoding='utf-8')

        if len(wronglabel_list) != 0:
            wronglabel_list.sort()
            df = pd.DataFrame(wronglabel_list, columns=['wrong label files'])
            df.to_csv(wronglabel_file, index=False, encoding='utf-8')

    def rename_file(self, src_path, log_file, job_count):
        img_list = os.listdir(os.path.join(src_path, 'images'))
        total = len(img_list)
        chunk_size = total // job_count
        chunk_list = micells.chunks_list(img_list, chunk_size)
        batches = [{'index': index, 'list': alist} for index, alist in enumerate(chunk_list)]
        log_lists = parmap.map(rename_file, batches, src_path=src_path, pm_pbar=True, pm_processes=job_count)

        log_list = []
        for log in log_lists:
            log_list = log_list + log

        log_list = sorted(log_list, key=lambda x: x[1])
        df = pd.DataFrame(log_list, columns=['original name', 'new name'])
        df.to_csv(log_file, index=False, encoding='utf-8')


if __name__ == '__main__':
    # startTime = time.time()
    # type = 'train'
    # # src_path = f'/home2/channelbiome/DataHub/Test_Train/{type}'
    # # log_path = f'/home2/channelbiome/DataHub/Test_Train/{type}_rename.csv'
    # src_path = f'/home2/channelbiome/DataHub/NIA_2022_Final_processed/{type}'
    # log_path = f'/home2/channelbiome/DataHub/NIA_2022_Final_processed/{type}_rename.csv'
    # basedir = '/home2/channelbiome/DataHub/NIA_2022_Final_processed'
    # tool = Nia2022DataBulder(basedir)
    # print(f'================= Start rename files = {src_path} ===========================')
    # tool.rename_file(src_path, log_path, job_count=32)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # # src_path = '/home2/channelbiome/NIA_Data/2022/json_test'
    # src_path = '/home2/channelbiome/NIA_Data/2022/221219'
    # corrupt_file = '/home2/channelbiome/DataHub/NIA_2022_Final_processed/corrupt_files.csv'
    # malformat_file = '/home2/channelbiome/DataHub/NIA_2022_Final_processed/malformat_files.csv'
    # wronglabel_file = '/home2/channelbiome/DataHub/NIA_2022_Final_processed/wrong_label_files.csv'
    # basedir = '/home2/channelbiome/DataHub/NIA_2022_Final_processed'
    # tool = Nia2022DataBulder(basedir)
    # print(f'================= Start Check Image Corrupt = {src_path} ===========================')
    # tool.check_image_corrupt(src_path, corrupt_file, malformat_file, wronglabel_file,
    #                          job_count=32, display_error=True, label_check=True)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    startTime = time.time()
    # spec_file = '/home2/channelbiome/DataHub/test/train_spec.yaml'
    # basedir = '/home2/channelbiome/DataHub/test'
    # # spec_file = '/home2/channelbiome/DataHub/Test_Train/train_spec_test.yaml'
    # # basedir = '/home2/channelbiome/DataHub/Test_Train'
    basedir = sys.argv[1]
    spec_file = sys.argv[2]
    job_count = int(sys.argv[3])
    tool = Nia2022DataBulder(basedir)
    print(f'================= Start build data set from sped file = {spec_file} ===========================')
    tool.build_dataset_from_spec(spec_file, save_split=True, job_count=job_count)
    endTime = time.time()
    print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # # yaml_path = '/home2/channelbiome/DataHub/test/raw_spec_nia2022.yaml'
    # # img_path = '/home2/channelbiome/NIA_Data/test/2022'
    # # json_path = None
    # # # yaml_path = '/home2/channelbiome/DataHub/test/raw_spec_nia2021.yaml'
    # # # img_path = '/home2/channelbiome/NIA_Data/test/2021/images'
    # # # json_path = '/home2/channelbiome/NIA_Data/test/2021/jsons'
    # # basedir = '/home2/channelbiome/DataHub/test'
    # basedir = sys.argv[1]
    # yaml_path = sys.argv[2]
    # img_path = sys.argv[3]
    # json_path = sys.argv[4]
    # tool = Nia2022DataBulder(basedir)
    # print(f'================= Start build raw spec file = {img_path} ===========================')
    # tool.build_raw_spec_file(yaml_path, img_path, json_path)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # # src_path = '/home2/channelbiome/NIA_Data/2022/221219'
    # src_path = '/home2/channelbiome/NIA_Data/2022/json_test'
    # meta_path = '/home2/channelbiome/DataHub/yolov5/web/data/FoodMetaTable.csv'
    # basedir = '/home2/channelbiome/DataHub/NIA_2022_Final_processed'
    # tool = Nia2022DataBulder(basedir)
    # print(f'================= Start build nia 2022 json files src = {src_path} ===========================')
    # tool.build_nia2022json(src_path, meta_path, job_count=8)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # img_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Image'
    # json_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Json'
    # drop_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_drop'
    # out_file = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed/mismatch_files.csv'
    # basedir = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed'
    # tool = Nia2022DataBulder(basedir)
    # print(f'=================Start drop mismatch files===========================')
    # tool.drop_mismatching_nia2021(img_path, json_path, drop_path, out_file, job_count=32)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # startTime = time.time()
    # src_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Json'
    # # src_path = '/home2/channelbiome/NIA_Data/2021/to_be_changed'
    # tar_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Json_cvrt'
    # meta_path = '/home2/channelbiome/DataHub/yolov5/web/data/FoodMetaTable.csv'
    # basedir = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed'
    # tool = Nia2022DataBulder(basedir)
    # tool.cvrt_nia2021json(src_path, tar_path, meta_path, job_count=32)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # basedir = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed'
    # tool = Nia2022DataBulder(basedir)
    # src_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Json'
    # rep_file = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed/classname_in_mata.csv'
    # tool.rename_folder_name(src_path, rep_file, folder_mode='json')

    # basedir = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed'
    # tool = Nia2022DataBulder(basedir)
    # # src_path = '/home2/channelbiome/NIA_Data/2021/NIA_2021_Json'
    # src_path = '/home2/channelbiome/NIA_Data/2022/221204'
    # meta_path = '/home2/channelbiome/DataHub/yolov5/web/data/FoodMetaTable.csv'
    # # out_file = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed/nia2022_classname_non_meta.csv'
    # out_file = '/home2/channelbiome/DataHub/NIA_2022_Dec_processed/nia2021_classname_non_meta.csv'
    # tool.check_json_classname_from_meta(src_path, meta_path, out_file)
