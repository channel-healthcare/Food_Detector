import json
import sys
import os
import time
import datetime
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import micells
from functools import partial

jsonTemplate = {"Code Name": "xxx", "Name": "xxx", "W": "xxx", "H": "xxx", "File Format": "xxx",
                 "Cat 1": "01", "Cat 2": "02", "Cat 3": "03", "Cat 4": "xx",
                 "Annotation Type": "binding", "Point(x,y)": "xxx", "Label": "0",
                 "Serving Size": "xx", "Camera Angle": "xx", "Cardinal Angle": "xx",
                 "Color of Container": "xx", "Material of Container": "xx",
                 "Illuminance": "xx", "Meta File": "xx"}

'''
python multiprocessing functions to be called
'''

def copy_files(filelist, img_dir, label_dir, json_dir, desc):
    pid = multiprocessing.current_process()
    for imagepath in tqdm(filelist, desc=f'{desc}, pid = {pid}'):
        shutil.copy(imagepath, img_dir)
        labelpath = imagepath.replace('.jpg', '.txt')
        shutil.copy(labelpath, label_dir)
        jsonpath = imagepath.replace('.jpg', '.json')
        shutil.copy(jsonpath, json_dir)

class YoloCVAT_v2:
    def __init__(self, basedir):
        self.basedir = basedir
        self.errlog = open(os.path.join(basedir, 'errorlog.txt'), 'w')

    def __del__(self):
        self.errlog.close()

    '''
    Build specification file(raw_spec.json) from image and label folders which is composed of .jpg and .json.
    Referfing to raw_spec.json, you can edit another specification file(train_spec.yaml) which will be utilized for 
    building data.yaml and train/valid/test data folder 
    '''

    def build_raw_spec_file(self, ori_path, create_json=False):
        print('start raw_spec file...')
        outfile = {}
        name_dic = {}
        jpg_total_cnt = 0
        txt_total_cnt = 0
        instance_total_cnt = 0
        directories = []
        for r, d, f in tqdm(list(os.walk(ori_path)), desc=f'building raw_spec file'):
            if r == ori_path:
                directories.extend(d)
            else:
                class_name = os.path.split(r)[1]
                jpg_cnt, txt_cnt = 0, 0

                image_files = [image_file for image_file in f if '.jpg' in image_file]
                jpg_cnt += len(image_files)
                if jpg_cnt == 0:
                    msg = f'image file are not exist. {class_name}'
                    print(msg)
                    self.errlog.write(msg + '\n')
                    continue

                for aimage in image_files:
                    txt_path = os.path.join(ori_path, r, aimage.replace('.jpg', '.txt'))
                    if not os.path.exists(txt_path):
                        msg = f'image file no exist, file = {txt_path}'
                        print(msg)
                        self.errlog.write(msg + '\n')
                        continue

                    if create_json:
                        instance_total_cnt += self.write_json(txt_path, class_name)

                    txt_cnt += 1

                name_dic[class_name] = txt_cnt
                jpg_total_cnt += jpg_cnt
                txt_total_cnt += txt_cnt

        outfile['directory count'] = len(directories)
        outfile['total txt file count'] = txt_total_cnt
        outfile['total jpg file count'] = jpg_total_cnt
        outfile['total instance count'] = instance_total_cnt
        outfile['class count'] = len(name_dic)
        outfile['directories'] = directories
        outfile['classes'] = name_dic

        with open(os.path.join(self.basedir, 'raw_spec.json'), 'w', encoding='utf-8') as f:
            json.dump(outfile, f, indent=2, ensure_ascii=False)

        print(f'build raw_spec.json done!! {os.path.join(self.basedir, "raw_spec.json")}')

    def write_json(self, txt_path, class_name):
        jsonlist = []
        line_count = 0
        with open(txt_path, 'r') as pf:
            lines = pf.readlines()
            line_count = len(lines)
            for aline in lines:
                contents = aline.strip().split(' ')
                jsonTemplate["Code Name"] = os.path.split(txt_path)[1].replace('.txt', '.jpg')
                jsonTemplate["Name"] = class_name
                jsonTemplate["Point(x,y)"] = f'{contents[1]},{contents[2]}'
                jsonTemplate["W"] = contents[3]
                jsonTemplate["H"] = contents[4]
                jsonlist.append(jsonTemplate)

        head_tail = os.path.split(txt_path)
        with open(os.path.join(head_tail[0], head_tail[1].replace('.txt', '.json')), 'w', encoding='utf-8') as f:
            json.dump(jsonlist, f, indent=2, ensure_ascii=False)

        return line_count

    '''
    Referring to train_spec.yaml, build data.yaml and train/valid/test data folder.
    # 1. build a list composed of filename , and edit *.txt from class 0 to class index
    # 2. train/val/test split
    # 3. copy files from images/json folder to image/label of train/val/test folder
    # 4. create data.yaml file.
    '''
    def build_dataset_from_spec(self, spec_file, src_path, multi_processing=False, multi_threshold = 80):
        with open(spec_file, 'r') as f:
            yalm_data = yaml.load(f, Loader=yaml.FullLoader)

        namelist = yalm_data['names']

        # 1.
        print('collecting images and labels')
        listFile = []
        for r, d, f in tqdm(list(os.walk(src_path)), desc=f'Collecting path information and change class id {src_path}'):
            if r == src_path:
                continue

            class_name = os.path.split(r)[1]
            image_files = [image_file for image_file in f if '.jpg' in image_file]
            for aimage in image_files:
                txt_path = os.path.join(src_path, r, aimage.replace('.jpg', '.txt'))
                if not os.path.exists(txt_path):
                    msg = f'image file no exist, file = {txt_path}'
                    print(msg)
                    self.errlog.write(msg + '\n')
                    continue

                new_lines = []
                with open(txt_path, 'r') as pf:
                    lines = pf.readlines()

                    for aline in lines:
                        contents = aline.split(' ')
                        contents[0] = str(namelist.index(class_name))
                        new_lines.append(' '.join(contents))

                with open(txt_path, 'w') as pf:
                    pf.writelines(new_lines)

                img_path = os.path.join(src_path, r, aimage)
                listFile.append(img_path)

        # 2.
        train_files, test_files = train_test_split(listFile, train_size=yalm_data['data_ratio']['train'],
                                                   random_state=42, shuffle=True)
        ratio = yalm_data['data_ratio']['val'] / (
                    yalm_data['data_ratio']['test'] + yalm_data['data_ratio']['val'])
        val_files, test_files = train_test_split(test_files, test_size=ratio, random_state=42, shuffle=True)

        print(f'total = {len(listFile)}, train = {len(train_files)}, val = {len(val_files)}, test = {len(test_files)}')

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

        # 3.
        print('start copy')
        copy_done_train, copy_done_val, copy_done_test = False, False, False
        if multi_processing:
            total = len(train_files)
            if total > multi_threshold:
                self._copy_files_multiprocessing(train_files, train_images, train_labels, train_jsons, 'train data processing')
                print('train data copy done!')
                copy_done_train = True
            total = len(val_files)
            if total > multi_threshold:
                self._copy_files_multiprocessing(val_files, val_images, val_labels, val_jsons, 'valid data processing')
                print('valid data copy done!')
                copy_done_val = True
            total = len(test_files)
            if total > multi_threshold:
                self._copy_files_multiprocessing(test_files, test_images, test_labels, test_jsons, 'test data processing')
                print('test data copy done!')
                copy_done_test = True

        if not copy_done_train:
            copy_files(train_files, train_images, train_labels, train_jsons, 'train data processing')
            print('train data copy done!')

        if not copy_done_train:
            copy_files(val_files, val_images, val_labels, val_jsons, 'valid data processing')
            print('valid data copy done!')

        if not copy_done_train:
            copy_files(test_files, test_images, test_labels, test_jsons, 'test data processing')
            print('test data copy done!')

        # 4.
        yalm_out = dict()
        yalm_out['path'] = os.path.abspath(basedir)
        yalm_out['train'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['train']))
        yalm_out['val'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['val']))
        yalm_out['test'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['test']))
        yalm_out['nc'] = yalm_data['nc']
        yalm_out['names'] = yalm_data['names']

        with open(os.path.join(basedir, 'data.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(yalm_out, f, default_flow_style=False, allow_unicode=True)

        print(f'build train finished!!')

    def _copy_files_multiprocessing(self, filelist, img_dir, label_dir, json_dir, desc):
        job_number = multiprocessing.cpu_count()
        total = len(filelist)
        chunk_size = total // job_number
        batches = micells.chunks_list(filelist, chunk_size)
        pool = Pool(job_number)
        copy_func = partial(copy_files, img_dir=img_dir, label_dir=label_dir, json_dir=json_dir, desc=desc)
        pool.map(copy_func, batches)
        print('Pool Ready')
        pool.close()
        pool.join()
        print('Pool Done')

    def statistics(self, spec_path):
        with open(spec_path, 'r') as jf:
            spec_file = json.load(jf)

        classes = spec_file['classes']
        target = pd.DataFrame(classes.values())
        stat = target.describe()
        print(stat)



if __name__ == '__main__':
    # /home2/channelbiome/NIA_Data/2022/221011-221014 /home2/channelbiome/DataHub/NIA_20221014_processed
    # startTime = time.time()
    # ori_path = sys.argv[1]
    # tar_path = sys.argv[2]
    # tool = YoloCVAT_v2(tar_path)
    # tool.build_raw_spec_file(ori_path, create_json=True)
    # endTime = time.time()
    # print(f'Elapsed Time = {datetime.timedelta(seconds=endTime - startTime)}')

    # /home2/channelbiome/DataHub/send_nia/train_spec.yaml /home2/channelbiome/NIA_Data/2022/221011 /home2/channelbiome/DataHub/send_nia
    # /home2/channelbiome/DataHub/NIA_20221014_processed/train_spec.yaml /home2/channelbiome/NIA_Data/2022/221011-221014 /home2/channelbiome/DataHub/NIA_20221014_processed
    # startTime = time.time()
    # spec_file = sys.argv[1]
    # src_path = sys.argv[2]
    # tar_path = sys.argv[3]
    # tool = YoloCVAT_v2(tar_path)
    # tool.build_dataset_from_spec(spec_file, src_path, multi_processing=True)
    # endTime = time.time()
    # print(f'Elapsed Time = {datetime.timedelta(seconds=endTime - startTime)}')

    # /home2/channelbiome/DataHub/NIA_20221014_processed/raw_spec.json /home2/channelbiome/DataHub/NIA_20221014_processed
    # /home2/channelbiome/DataHub/NIA_20221011_processed/raw_spec.json /home2/channelbiome/DataHub/NIA_20221011_processed
    startTime = time.time()
    spec_file = sys.argv[1]
    dest_path = sys.argv[2]
    tool = YoloCVAT_v2(dest_path)
    tool.statistics(spec_file)
    endTime = time.time()
    print(f'Elapsed Time = {datetime.timedelta(seconds=endTime - startTime)}')
