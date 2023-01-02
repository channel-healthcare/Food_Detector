import json
import argparse
import os
import time
import datetime
import shutil
from sklearn.model_selection import train_test_split
from configparser import ConfigParser
import yaml
from tqdm import tqdm


class YoloDataProcessor:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read('data/config.ini')
        self.errlog = open(os.path.join(self.config.get('path', 'basedir'), 'errorlog.txt'), 'w')

    def __del__(self):
        self.errlog.close()

    '''
    Build specification file(raw_spec.json) from image and label folders which is composed of .jpg and .json.
    Referfing to raw_spec.json, you can edit another specification file(train_spec.yaml) which will be utilized for 
    building data.yaml and train/valid/test data folder 
    '''
    def build_raw_spec_file(self):
        print('start raw_spec file...')
        outfile = {}
        name_dic = {}
        root = self.config.get('path', 'json_dir')
        dir_cnt = 0
        match_total_cnt = 0
        json_total_cnt = 0
        image_root = self.config.get('path', 'image_dir')
        for r, d, f in os.walk(root):
            if r == root:
                outfile['directory count'] = len(d)
                outfile['directory'] = d

            if ' json' in r:
                afile_name_set = set()
                dir_cnt += 1
                file_cnt = 0
                image_dir = os.path.split(r)[1].replace(' json', '')
                if len(f) == 0:
                    print(f'blank directory = {r}')
                for afile in tqdm(f, desc=f'{dir_cnt}/{outfile["directory count"]} processing {r}'):
                    try:
                        json_total_cnt += 1
                        image_path = os.path.join(image_root, image_dir, afile.replace('.json', '.jpg'))
                        if not os.path.exists(image_path):
                            msg = f'image file no exist, file = {image_path}'
                            self.errlog.write(msg + '\n')
                            continue

                        filename = os.path.join(r, afile)
                        with open(filename, 'r') as jf:
                            json_data = json.load(jf)

                        for aDic in json_data:
                            afile_name_set.add(aDic['Name'])
                            if aDic['Name'] in name_dic:
                                name_dic[aDic['Name']] += 1
                            else:
                                name_dic[aDic['Name']] = 1
                        file_cnt += 1
                        match_total_cnt += 1
                    except Exception as e:
                        msg = f'exception = {e}, dir = {r}, filename = {afile}'
                        self.errlog.write(msg + '\n')
                        print(msg)

                classfolder = os.path.split(r)[1]
                outfile[f'{classfolder.replace(" json", "")}'] = file_cnt
                outfile[f'{classfolder}'] = list(afile_name_set)

        outfile['total json file count'] = json_total_cnt
        outfile['total matching file count'] = match_total_cnt
        outfile['class count'] = len(name_dic)
        outfile['classes'] = name_dic

        with open(self.config.get('path', 'raw_spec'), 'w', encoding='utf-8') as f:
            json.dump(outfile, f, indent=2, ensure_ascii=False)

        print(f'build raw_spec.json done!! directory num = {outfile["classes"]}')

    '''
    Refering to train_spec.yaml, build data.yaml and train/valid/test data folder.
    # 1. build a list composed of filename and label.txt formed data from json folder until count number
    # 2. train/val/test split
    # 3. copy files from images/json folder to image/label of train/val/test folder
    # 4. create data.yaml file.
    '''
    def build_dataset_from_spec(self, ignore_total=False, alias_naming=False):
        with open(self.config.get('path', 'train_spec'), 'r') as f:
            yalm_data = yaml.load(f, Loader=yaml.FullLoader)

        if self.check_train_spec_file(yalm_data) == False and ignore_total == False:
            print('fail build train data')
            return

        print(f'start build train. data size = {yalm_data["data_total"]}')
        namelist = yalm_data['names']

        # 1.
        listFile = []
        json_root = self.config.get('path', 'json_dir')
        image_root = self.config.get('path', 'image_dir')
        dicLabel = yalm_data['label']
        # key: directory value: dictionary composed of 'class' and 'count'
        for key, value in dicLabel.items():
            json_dir = os.path.join(json_root, key + ' json')
            json_files = os.listdir(json_dir)
            className = value['class']
            count = value['count']
            success_cnt = 0
            for file in json_files:
                image_path = os.path.join(image_root, key, file.replace('.json', '.jpg'))
                if not os.path.exists(image_path):
                    msg = f'image file no exist, file = {image_path}'
                    print(msg)
                    self.errlog.write(msg + '\n')
                    continue

                json_filename = os.path.join(json_dir, file)
                with open(json_filename, 'r') as f:
                    json_data = json.load(f)

                lableContents = ''
                # multiple labelling.
                for adic in json_data:
                    aClass = adic['Name']
                    if aClass not in namelist:
                        msg = f'{aClass} is not in namelist, file = {json_filename}'
                        print(msg)
                        self.errlog.write(msg + '\n')
                        continue

                    aClass = namelist.index(aClass)
                    center = adic['Point(x,y)']
                    center = center.split(',')
                    width = adic['W']
                    height = adic['H']
                    lableContents += f'{aClass} {center[0]} {center[1]} {width} {height}\n'

                lableContents = lableContents.rstrip()
                listFile.append(image_path + '|' + lableContents)

                success_cnt += 1
                if success_cnt >= count:
                    break
            if success_cnt < count:
                msg = f'{key} count = {success_cnt}'
                print(msg)
                self.errlog.write(msg + '\n')
        # return

        # 2.
        train_files, test_files = train_test_split(listFile, train_size=yalm_data['data_ratio']['train'],
                                                   random_state=42, shuffle=True)
        ratio = yalm_data['data_ratio']['val'] / (yalm_data['data_ratio']['test'] + yalm_data['data_ratio']['val'])
        val_files, test_files = train_test_split(test_files, test_size=ratio, random_state=42, shuffle=True)

        basedir = self.config.get('path', 'basedir')
        train_images = os.path.join(basedir, yalm_data['folder']['train'], 'images')
        train_labels = os.path.join(basedir, yalm_data['folder']['train'], 'labels')
        val_images = os.path.join(basedir, yalm_data['folder']['val'], 'images')
        val_labels = os.path.join(basedir, yalm_data['folder']['val'], 'labels')
        test_images = os.path.join(basedir, yalm_data['folder']['test'], 'images')
        test_labels = os.path.join(basedir, yalm_data['folder']['test'], 'labels')

        if not os.path.exists(train_images): os.makedirs(train_images)
        if not os.path.exists(train_labels): os.makedirs(train_labels)
        if not os.path.exists(val_images): os.makedirs(val_images)
        if not os.path.exists(val_labels): os.makedirs(val_labels)
        if not os.path.exists(test_images): os.makedirs(test_images)
        if not os.path.exists(test_labels): os.makedirs(test_labels)

        # 3.
        for afile in tqdm(train_files, desc='train data processing'):
            imagepath, label = afile.split('|')
            shutil.copy(imagepath, train_images)
            labelpath = os.path.join(train_labels, os.path.split(imagepath)[1].replace('.jpg', '.txt'))
            with open(labelpath, 'w') as f:
                f.write(label)

        for afile in tqdm(val_files, desc='valid data processing'):
            imagepath, label = afile.split('|')
            shutil.copy(imagepath, val_images)
            labelpath = os.path.join(val_labels, os.path.split(imagepath)[1].replace('.jpg', '.txt'))
            with open(labelpath, 'w') as f:
                f.write(label)

        for afile in tqdm(test_files, desc='test data processing'):
            imagepath, label = afile.split('|')
            shutil.copy(imagepath, test_images)
            labelpath = os.path.join(test_labels, os.path.split(imagepath)[1].replace('.jpg', '.txt'))
            with open(labelpath, 'w') as f:
                f.write(label)

        # 4.
        yalm_out = dict()
        yalm_out['path'] = os.path.abspath(basedir)
        yalm_out['train'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['train']))
        yalm_out['val'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['val']))
        yalm_out['test'] = os.path.abspath(os.path.join(basedir, yalm_data['folder']['test']))
        yalm_out['nc'] = yalm_data['nc']
        if alias_naming:
            yalm_out['names'] = yalm_data['alias']
        else:
            yalm_out['names'] = yalm_data['names']
        with open(os.path.join(basedir, 'data.yaml'), 'w', encoding='utf-8') as f:
            yaml.dump(yalm_out, f, default_flow_style=False, allow_unicode=True)

        print(f'build train finished!!')

    '''
    fix wrong labeled .json files, replacing duplicated food class names with just one. 
    '''
    def fix_labelfiles(self):
        print('Start fix wrong label files')
        json_file = self.config.get('path', 'raw_spec')
        with open(json_file, 'r') as jf:
            json_data = json.load(jf)

        json_root = self.config.get('path', 'json_dir')
        deposit = os.path.join(json_root, 'deposit')
        if not os.path.exists(deposit): os.makedirs(deposit)
        change_cnt = 0
        folders = json_data['directory']
        for folder in folders:
            classlist = json_data[folder]
            # if class name is more than one, change the name with most frequent one.
            if len(classlist) != 1:
                # dicClass = {k: json_data['classes'][k] for k in classlist}
                # dicClass = dict(sorted(dicClass.items(), key=lambda item: item[1], reverse=True))
                classname = sorted(classlist, key=lambda key: json_data['classes'][key], reverse=True)[0]
                json_dir = os.path.join(json_root, folder)
                json_files = os.listdir(json_dir)
                for ajson in json_files:
                    label_path = os.path.join(json_dir, ajson)
                    if os.path.splitext(label_path)[1] != '.json':
                        continue

                    with open(label_path, 'r') as f:
                        label_data = json.load(f)

                    change_name = False
                    for adic in label_data:
                        if adic['Name'] != classname:
                            change_name = True

                    if change_name:
                        newname_list = []
                        for adic in label_data:
                            newname_list.append(adic['Name'])
                            adic['Name'] = classname
                        # copy original label file
                        shutil.copy(label_path, deposit)
                        adic['Name'] = classname
                        # change class name and save the contents.
                        with open(label_path, 'w') as f:
                            json.dump(label_data, f, indent=2)

                        msg = f'label name changed from {newname_list}, to {classname}, file = {label_path}'
                        print(msg)
                        self.errlog.write(msg + '\n')
                        change_cnt += 1

        print(f'End of change label files, changed files counts = {change_cnt}')

    '''
    check class name is duplicated and only print it on console.
    '''
    def check_dupl_classname(self):
        json_file = self.config.get('path', 'raw_spec')
        with open(json_file, 'r') as jf:
            json_data = json.load(jf)

        dirs = json_data['directory']
        dicClass = {}
        for dir in dirs:
            classlist = json_data[dir]
            for aclass in classlist:
                if dicClass.get(aclass) == None:
                    dicClass[aclass] = 1
                else:
                    dicClass[aclass] += 1

        for item in dicClass.items():
            if item[1] != 1:
                print(f'{item[0]}: {item[1]}')

    '''
    Nothing.
    '''
    def probe_label(self, dir_path):
        jsonfiles = os.listdir(dir_path)
        dicClass = dict()
        for jsonfile in jsonfiles:
            with open(os.path.join(dir_path, jsonfile)) as f:
                json_data = json.load(f)

            for adic in json_data:
                classname = adic['Name']
                if dicClass.get(classname) == None:
                    dicClass[classname] = 1
                else:
                    dicClass[classname] += 1

        print(dicClass)

    '''
    check train_spec.yaml file is written right or not.
    this function can be utilized by itself or before building data.yaml and data folders lastly.
    '''
    def check_train_spec_file(self, yalm_data=None):
        NoError = True
        if yalm_data == None:
            with open(self.config.get('path', 'train_spec'), 'r') as f:
                yalm_data = yaml.load(f, Loader=yaml.FullLoader)

        train_ratio = yalm_data['data_ratio']['train']
        val_ratio = yalm_data['data_ratio']['val']
        test_ratio = yalm_data['data_ratio']['test']

        if train_ratio + val_ratio + test_ratio != 1:
            msg = f'train/val/test ratio is not acceptable {train_ratio}/{val_ratio}/{test_ratio}'
            print(msg)
            self.errlog.write(msg + '\n')
            NoError = False

        with open(self.config.get('path', 'raw_spec'), 'r') as f:
            json_data = json.load(f)

        namelist = yalm_data['names']
        labels = yalm_data['label']
        total_images = 0
        for key, value in labels.items():
            classname = value['class']
            if not classname in namelist:
                msg = f'{classname} is not classname'
                print(msg)
                self.errlog.write(msg + '\n')
                NoError = False

            if not classname in json_data['classes'].keys():
                msg = f'{classname} is not classname of raw_spec.json'
                print(msg)
                self.errlog.write(msg + '\n')
                NoError = False

            image_count = value['count']
            if image_count > json_data[key]:
                msg = f'{key} is not enough. req = {image_count} cap = {json_data[key]} '
                print(msg)
                self.errlog.write(msg + '\n')
                NoError = False


            total_images += value['count']

        if yalm_data['data_total'] != total_images:
            msg = f'summation of labels file is {total_images}, whereas data_total is {yalm_data["data_total"]}'
            print(msg)
            self.errlog.write(msg + '\n')
            NoError = False

        return NoError



if __name__ == '__main__':
    startTime = time.time()
    parser = argparse.ArgumentParser(description='receive the parameters')
    parser.add_argument('--dir', type=str, help='type: file path')
    args = parser.parse_args()
    tool = YoloDataProcessor()
    tool.build_raw_spec_file()
    # tool.build_dataset_from_spec(ignore_total=True, alias_naming=True)
    # tool.probe_label('../../food_image/Json/고구마 json')
    # tool.check_dupl_classname()
    # tool.fix_labelfiles()
    # tool.check_train_spec_file()
    endTime = time.time()
    print(f'Elapsed Time = {datetime.timedelta(seconds=endTime - startTime)}')

    # command = 'unzip -n -j ../../food_image/Image.zip */A240201_50006.jpg -d ../Foods/train'
    # # command = 'unzip -O cp949 -j ../../food_image/Image.zip 가래떡/A240201_50001.jpg -d ../Foods/train'
    # os.system(command)
