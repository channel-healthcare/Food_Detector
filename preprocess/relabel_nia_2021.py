import json
import sys
import os
import time
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
import collections
import multiprocessing
from multiprocessing import Pool
from functools import partial
from preprocess.micells import chunks_list
import csv

def collect_files_info(abatch, log_path):
    pos = abatch['index']
    json_total_cnt, over_cnt, double_named_file = 0, 0, 0
    name_dic = {}
    exception_files = []
    dir_name_info = []
    double_named = []
    for dir in tqdm(abatch['list'], desc=f'Collecting information in folders. Index = {pos}'):
        afile_name_set = set()
        file_cnt = 0
        folder_name = os.path.split(dir)[1]
        filelist = os.listdir(dir)
        for afile in filelist:
            try:
                filename = os.path.join(dir, afile)
                with open(filename, 'r') as jf:
                    json_data = json.load(jf)

                for aDic in json_data:
                    afile_name_set.add(aDic['Name'])
                    if aDic['Name'] in name_dic:
                        name_dic[aDic['Name']] += 1
                    else:
                        name_dic[aDic['Name']] = 1

                    if float(aDic['H']) < 0 or float(aDic['H']) > 1 or \
                            float(aDic['W']) < 0 or float(aDic['W']) > 1:
                        over_cnt += 1

                    json_total_cnt += 1
                    file_cnt += 1
            except Exception as e:
                exception_files.append(filename)
                os.system(f'rm "{filename}"')
                msg = f'exception = {e}, dir = {dir}, filename = {afile}'
                print(msg)


        aclass = folder_name.replace(' json', '')
        dir_name_info.append({aclass: file_cnt})
        dir_name_info.append({f'{aclass} names': {f'{name}': name_dic[name] for name in afile_name_set}})
        if len(afile_name_set) > 1:
            double_named.append(aclass)
            double_named_file += file_cnt

    # directory name, class name, folder name info, exception file, files count,
    outfile = {}
    outfile['total json files count'] = json_total_cnt
    outfile['directories'] = [os.path.split(name)[1].replace(' json', '') for name in abatch['list']]
    outfile['classes'] = name_dic
    outfile['folder name info'] = dir_name_info
    outfile['Width/Height overflow files count'] = over_cnt
    outfile['exception files'] = exception_files
    outfile['double named'] = double_named
    outfile['double named file'] = double_named_file
    return outfile

def relabel_folders(abatch, tar_path, log_path):
    pos = abatch['index']
    # pid = multiprocessing.current_process()
    relabel_log = []
    for dir in tqdm(abatch['list'], desc=f'relabeling in folders. Index = {pos}'):
        folder_name = os.path.split(dir)[1]
        filelist = os.listdir(dir)
        for afile in filelist:
            try:
                filename = os.path.join(dir, afile)
                with open(filename, 'r') as jf:
                    json_data = json.load(jf)

                class_name = folder_name.split(' ')[0]  # remove ' json' string
                for aDic in json_data:
                    alog = {}
                    over_flag = False
                    alog['file name'] = afile
                    if aDic['Name'] != class_name:
                        alog['class name'] = f"{aDic['Name']} -> {class_name}"
                        aDic['Name'] = class_name
                    if float(aDic['H']) < 0:
                        alog['H over'] = aDic['H']
                        aDic['H'] = '0'
                    elif float(aDic['H']) > 1:
                        alog['H over'] = aDic['H']
                        aDic['H'] = '1'
                    if float(aDic['W']) < 0:
                        alog['W over'] = aDic['W']
                        aDic['W'] = '0'
                    elif float(aDic['W']) > 1:
                        alog['W over'] = aDic['W']
                        aDic['W'] = '1'

                    if alog:
                        relabel_log.append(alog)

                    apath = os.path.join(tar_path, folder_name)
                    if not os.path.exists(apath):
                        os.makedirs(apath)
                    savename = os.path.join(apath, afile)
                    with open(savename, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                msg = f'exception = {e}, dir = {dir}, filename = {afile}'
                print(msg)

    df = pd.DataFrame.from_dict(relabel_log)
    df.to_csv(os.path.join(log_path, f'relabel_log_{pos}.csv'), index=False, header=True)

class ReLabeler(object):
    def __init__(self, basedir):
        self.basedir = basedir
        self.errlog = open(os.path.join(basedir, 'errorlog.txt'), 'w')

    def __del__(self):
        self.errlog.close()

    def build_raw_spec_file(self, root):
        print('start raw_spec file...')
        outfile = {}
        name_dic = {}
        dir_name_info = {}
        dir_cnt = json_total_cnt = over_cnt = 0
        dir_names = exception_files = []
        for r, d, f in tqdm(list(os.walk(root)), desc=f'Raw Spec Processing of {root}', leave=True):
            if r == root:
                dir_names = d

            if os.path.split(r)[1] in dir_names:
                afile_name_set = set()
                dir_cnt += 1
                file_cnt = 0
                if len(f) == 0:
                    print(f'blank directory = {r}')
                # for afile in tqdm(f, desc=f'{dir_cnt}/{outfile["directory count"]} processing {r}', leave=True):
                for afile in f:
                    try:
                        filename = os.path.join(r, afile)
                        with open(filename, 'r') as jf:
                            json_data = json.load(jf)

                        for aDic in json_data:
                            afile_name_set.add(aDic['Name'])
                            if aDic['Name'] in name_dic:
                                name_dic[aDic['Name']] += 1
                            else:
                                name_dic[aDic['Name']] = 1

                            if float(aDic['H']) < 0 or float(aDic['H']) > 1 or \
                                    float(aDic['W']) < 0 or float(aDic['W']) > 1:
                                over_cnt += 1

                        json_total_cnt += 1
                        file_cnt += 1
                    except Exception as e:
                        exception_files.append(filename)
                        os.system(f'rm "{filename}"')
                        msg = f'exception = {e}, dir = {r}, filename = {afile}'
                        self.errlog.write(msg + '\n')
                        print(msg)

                class_folder = os.path.split(r)[1]
                class_name = class_folder.replace(" json", "")
                dir_name_info[f'{class_name}'] = file_cnt
                dicClassNames = {aclass: name_dic[aclass] for aclass in afile_name_set}
                dir_name_info[f'{class_name} names'] = dicClassNames

        outfile['directory count'] = len(dir_names)
        outfile['class count'] = len(name_dic)
        outfile['total json files count'] = json_total_cnt
        outfile['Width/Height overflow files count'] = over_cnt
        outfile['exception files count'] = len(exception_files)
        outfile['directories'] = dir_names
        outfile['classes'] = name_dic
        outfile['folder name info'] = collections.OrderedDict(sorted(dir_name_info.items()))
        outfile['exception files'] = exception_files

        saved_file = os.path.join(self.basedir, 'nia-2021-raw_spec.json')
        with open(saved_file, 'w', encoding='utf-8') as f:
            json.dump(outfile, f, indent=2, ensure_ascii=False)

        print(f'saved spec file {saved_file}')

    def build_raw_spec_file_mutiprocess(self, root, cpu_count=multiprocessing.cpu_count()):
        for r, d, f in tqdm(list(os.walk(root)), desc=f'Collecting paths of {root}', leave=True):
            if r == root:
                folder_list = [os.path.join(r, folder) for folder in d]
                break

        job_number = cpu_count
        total = len(folder_list)
        chunk_size = total // job_number
        chunk_list = chunks_list(folder_list, chunk_size)
        batches = [{'index': index, 'list': alist} for index, alist in enumerate(chunk_list)]
        pool = Pool(job_number)
        copy_func = partial(collect_files_info, log_path=self.basedir)
        spec_list = pool.map(copy_func, batches)
        print('Pool Ready')
        pool.close()
        pool.join()
        print('Pool Done')

        directories, exceptions, double_named = [], [], []
        folder_name_info ,classes = {}, {}
        total_json_files, over_count, double_named_file = 0, 0, 0
        for spec in spec_list:
            directories.extend(spec['directories'])
            exceptions.extend(spec['exception files'])
            double_named.extend(spec['double named'])
            double_named_file += spec['double named file']
            total_json_files += spec['total json files count']
            over_count += spec['Width/Height overflow files count']
            classes.update(spec['classes'])
            for adic in spec['folder name info']:
                folder_name_info.update(adic)

        outfile = {}
        outfile['directory count'] = len(directories)
        outfile['class count'] = len(classes)
        outfile['total json files count'] = total_json_files
        outfile['Width/Height overflow files count'] = over_count
        outfile['double named class count'] = len(double_named)
        outfile['double named file count'] = double_named_file
        outfile['exception files count'] = len(exceptions)
        outfile['directories'] = directories
        outfile['classes'] = classes
        outfile['folder name info'] = collections.OrderedDict(folder_name_info.items())
        outfile['double named class'] = sorted(double_named)
        outfile['exception files'] = exceptions

        saved_file = os.path.join(self.basedir, 'nia-2021-raw_spec.json')
        with open(saved_file, 'w', encoding='utf-8') as f:
            json.dump(outfile, f, indent=2, ensure_ascii=False)

        print(f'saved spec file {saved_file}')

    def relabel(self, src_path, tar_path):
        relabel_log, folder_list = [], []
        for r, d, f in tqdm(list(os.walk(src_path)), desc=f'ReLabel Processing of {src_path}', leave=True):
            folder_name = os.path.split(r)[1]
            if r == src_path:
                folder_list.extend(d)
            elif folder_name in folder_list:
                for afile in f:
                    try:
                        filename = os.path.join(r, afile)
                        with open(filename, 'r') as jf:
                            json_data = json.load(jf)

                        class_name = folder_name.split(' ')[0]  # remove ' json' string
                        for aDic in json_data:
                            alog = {}
                            alog['file name'] = afile
                            if aDic['Name'] != class_name:
                                alog['class name'] = f"{aDic['Name']} -> {class_name}"
                                aDic['Name'] = class_name
                            if float(aDic['H']) < 0:
                                alog['H over'] = aDic['H']
                                aDic['H'] = '0'
                            if float(aDic['H']) > 1:
                                alog['H over'] = aDic['H']
                                aDic['H'] = '1'
                            if float(aDic['W']) < 0:
                                alog['W over'] = aDic['W']
                                aDic['W'] = '0'
                            if float(aDic['W']) > 1:
                                alog['W over'] = aDic['W']
                                aDic['W'] = '1'

                            if alog:
                                relabel_log.append(alog)

                            apath = os.path.join(tar_path, folder_name)
                            if not os.path.exists(apath):
                                os.makedirs(apath)
                            savename = os.path.join(apath, afile)
                            with open(savename, 'w', encoding='utf-8') as f:
                                json.dump(json_data, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        msg = f'exception = {e}, dir = {r}, filename = {afile}'
                        self.errlog.write(msg + '\n')
                        print(msg)

        df = pd.DataFrame.from_dict(relabel_log)
        df.to_csv(os.path.join(self.basedir, 'relabel_log.csv'), index=False, header=True)

    def relabel_mutiprocess(self, src_path, tar_path, cpu_count=multiprocessing.cpu_count()):
        for r, d, f in tqdm(list(os.walk(src_path)), desc=f'Collecting paths of {src_path}', leave=True):
            if r == src_path:
                folder_list = [os.path.join(r, folder) for folder in d]
                break

        job_number = cpu_count
        total = len(folder_list)
        chunk_size = total // job_number
        chunk_list = chunks_list(folder_list, chunk_size)
        batches = [{'index': index, 'list': alist} for index, alist in enumerate(chunk_list)]
        pool = Pool(job_number)
        copy_func = partial(relabel_folders, tar_path=tar_path, log_path=self.basedir)
        pool.map(copy_func, batches)
        print('Pool Ready')
        pool.close()
        pool.join()
        print('Pool Done')

        with open(os.path.join(self.basedir, 'relabel_summary.csv'), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['relabel_log files information'])
            for adic in batches:
                alist = adic['list']
                alist.insert(0, f'relabel_log_{adic["index"]}.csv paths')
                writer.writerow(alist)

        print(f'relabel done!')

if __name__ == '__main__':
    # startTime = time.time()
    # tar_path = sys.argv[1]
    # log_path = sys.argv[2]
    # tool = ReLabeler(log_path)
    # tool.build_raw_spec_file(tar_path)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # /home2/channelbiome/food_image/spec_test /home2/channelbiome/food_image/spec_test_relabel /home2/channelbiome/DataHub/yolov5/docs/NIA_2021
    # startTime = time.time()
    # src_path = sys.argv[1]
    # tar_path = sys.argv[2]
    # log_path = sys.argv[3]
    # tool = ReLabeler(log_path)
    # tool.relabel(src_path, tar_path)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')

    # /home2/channelbiome/food_image/NIA_2021_Json /home2/channelbiome/DataHub/yolov5/docs/NIA_2021
    # /home2/channelbiome/food_image/NIA_2021_Relabeled /home2/channelbiome/DataHub/yolov5/docs/NIA_2021
    # /home2/channelbiome/food_image/spec_test /home2/channelbiome/DataHub/yolov5/docs/NIA_2021
    startTime = time.time()
    tar_path = sys.argv[1]
    log_path = sys.argv[2]
    tool = ReLabeler(log_path)
    tool.build_raw_spec_file_mutiprocess(tar_path)
    endTime = time.time()
    print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')


    # /home2/channelbiome/food_image/spec_test /home2/channelbiome/food_image/spec_test_relabel /home2/channelbiome/DataHub/yolov5/docs/NIA_2021
    # /home2/channelbiome/food_image/NIA_2021_Json /home2/channelbiome/food_image/NIA_2021_Relabeled /home2/channelbiome/DataHub/yolov5/docs/NIA_2021
    # startTime = time.time()
    # src_path = sys.argv[1]
    # tar_path = sys.argv[2]
    # log_path = sys.argv[3]
    # tool = ReLabeler(log_path)
    # tool.relabel_mutiprocess(src_path, tar_path)
    # endTime = time.time()
    # print(f'Elapsed Time = {timedelta(seconds=endTime - startTime)}')
