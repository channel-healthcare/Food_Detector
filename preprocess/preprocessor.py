import json
import argparse
import os
import yaml
import time
import datetime
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

def replace_label(replace, errorlog):
    df = pd.read_csv(replace)
    dicOri2Repl = dict(zip(df['original'], df['replace']))
    # read all contents from errorlog file
    with open(errorlog, 'r') as ef:
        lines = ef.readlines()

    # replace key with value
    for aline in lines:
        for key, value in dicOri2Repl.items():
            if key in aline:
                # ex: build label error 'peach' is not in list, file = D:\GitHubDepo\yolov5\data\food\복숭아 json\A220120XX_10307.json
                filename = aline[aline.find('file = ') + 7:].rstrip()
                with open(filename, 'r') as jf:
                    json_data = json.load(jf)

                for aDic in json_data:
                    if aDic['Name'] == key:
                        aDic['Name'] = value

                with open(filename, 'w') as jf:
                    json.dump(json_data, jf, indent=2)

                print(f'replace {key} with {value}. file = {filename}')


class YOLOLabelBuilder:
    def __init__(self, basedir):
        self.basedir = basedir
        self.errlog = open(os.path.join(basedir, 'errorlog.txt'), 'w')


    def __del__(self):
        self.errlog.close()

    def load_data_yalm(self, filePath, drawContents = False):
        with open(filePath) as f:
            reader = yaml.load(f, Loader=yaml.FullLoader)
            root = reader['path']
            self.train_images = os.path.join(root, reader['train'])
            self.train_labels = os.path.join(os.path.split(self.train_images)[0], 'labels')
            self.val_images = os.path.join(root, reader['val'])
            self.val_labels = os.path.join(os.path.split(self.val_images)[0], 'labels')
            self.test_images = os.path.join(root, reader['test'])
            self.test_labels = os.path.join(os.path.split(self.test_images)[0], 'labels')
            self.data_class = reader['names']
            print(f'data class loaded from {filePath}')
            if drawContents:
                print(f'classes = {self.data_class}')

    def build_a_label(self, filePath):
        try:
            with open(filePath, 'r') as rf:
                json_data = json.load(rf)
                fname, ext = os.path.splitext(filePath)
                with open(f'{fname}.txt', 'w') as wf:
                    for aDic in json_data:
                        aClass = aDic['Name']
                        aClass = self.data_class.index(aClass)
                        center = aDic['Point(x,y)']
                        center = center.split(',')
                        width = aDic['W']
                        height = aDic['H']
                        wf.write(f'{aClass} {center[0]} {center[1]} {width} {height}')
            return True
        except Exception as e:
            msg = f'build label error {e}, file = {filePath}'
            self.errlog.write(msg + '\n')
            print(msg)
            return False

    def build_dir_label(self, dir):
        filelist = []
        for (r, d, f) in os.walk(dir):
            for afile in f:
                if os.path.splitext(afile)[1] == '.json':
                    filepath = os.path.join(r, afile)
                    if self.build_a_label(filepath):
                        filelist.append(filepath)

        with open(os.path.join(dir, 'buildlog.txt'), 'w') as f:
            f.write('\n'.join(filelist))

        print(f'build label {len(filelist)} files')

    def rearrage_data(self, dir, train=0.8, valid=0.1, test=0.1):
        imagelist = []
        for (r, d, f) in os.walk(dir):
            if 'json' in r: continue
            for afile in f:
                if '.jpg' in afile:
                    filepath = os.path.join(r, afile)
                    imagelist.append(filepath)


        train_files, test_files = train_test_split(imagelist, train_size=train, random_state=42, shuffle=True)
        val_files, test_files = train_test_split(test_files, test_size=0.5, random_state=42, shuffle=True)

        if not os.path.exists(self.train_images): os.makedirs(self.train_images)
        if not os.path.exists(self.train_labels): os.makedirs(self.train_labels)
        if not os.path.exists(self.val_images): os.makedirs(self.val_images)
        if not os.path.exists(self.val_labels): os.makedirs(self.val_labels)
        if not os.path.exists(self.test_images): os.makedirs(self.test_images)
        if not os.path.exists(self.test_labels): os.makedirs(self.test_labels)

        for train_f in train_files:
            head, tail = os.path.split(train_f)
            filename = os.path.splitext(tail)[0]
            labelfile = os.path.join(head+' json', filename+'.txt')
            if os.path.exists(labelfile):
                shutil.copy(train_f, self.train_images)
                shutil.copy(labelfile, self.train_labels)

        for val_f in val_files:
            head, tail = os.path.split(val_f)
            filename = os.path.splitext(tail)[0]
            labelfile = os.path.join(head+' json', filename+'.txt')
            if os.path.exists(labelfile):
                shutil.copy(val_f, self.val_images)
                shutil.copy(labelfile, self.val_labels)

        for test_f in test_files:
            head, tail = os.path.split(test_f)
            filename = os.path.splitext(tail)[0]
            labelfile = os.path.join(head+' json', filename+'.txt')
            if os.path.exists(labelfile):
                shutil.copy(test_f, self.test_images)
                shutil.copy(labelfile, self.test_labels)


if __name__ == '__main__':
    startTime = time.time()
    parser = argparse.ArgumentParser(description='receive the parameters')
    parser.add_argument('--file', type=str, help='type: file path')
    parser.add_argument('--dir', type=str, help='type: file path')
    parser.add_argument('--dataconf', type=str, help='type: file path')
    args = parser.parse_args()
    tool = YOLOLabelBuilder(args.dir)
    tool.load_data_yalm(args.dataconf, drawContents=True)
    tool.rearrage_data(args.dir)
    # tool.build_dir_label(args.dir)
    endTime = time.time()
    print(f'Elapsed Time = {datetime.timedelta(seconds=endTime-startTime)}')

    # replace_label("D:/GitHubDepo/yolov5/data/food/replace_label.csv", "D:/GitHubDepo/yolov5/data/food/errorlog.txt")
