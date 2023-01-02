========================================================================================================================
gpu watch....
watch -d -n 1 nvidia-smi

========================================================================================================================
이미지 파일 압축 해제 속도
360개 파일 Elapsed Time = 0:04:11.351849

========================================================================================================================
분할압축
zip -r out_file.zip src_dir
zip -s 4g out_file.zip --out out_part_file.zip

분할압축해제
zip -F out_part_file.zip --out join_file.zip
unzip join_file.zip

========================================================================================================================
train

yolov5s
python -m torch.distributed.run --nproc_per_node 2 train.py --data "../Foods5K/data.yaml" --epoch 3 --batch-size 128 --cfg "models/yolov5s.yaml" --weights yolov5s.pt --name F5K_Yolov5s --device 0,1

yolov5m
python -m torch.distributed.run --nproc_per_node 2 train.py --data "../Foods5K/data.yaml" --epoch 100 --batch-size 32 --cfg "models/yolov5m.yaml" --weights yolov5m.pt --name F5K_Yolov5m --device 0,1

yolov5l
python -m torch.distributed.run --nproc_per_node 2 train.py --data "../Foods20K/data.yaml" --epoch 100 --batch-size 32 --cfg "models/yolov5l.yaml" --weights yolov5l.pt --name F5K_Yolov5l --device 0,1 --patience 30

yolov5x
python -m torch.distributed.run --nproc_per_node 2 train.py --data "../Foods20K/data.yaml" --epoch 100 --batch-size 32 --cfg "models/yolov5x.yaml" --weights yolov5x.pt --name F5K_Yolov5x --device 0,1 --patience 30

python -m torch.distributed.run --nproc_per_node 2 train.py --data "data/food/data.yaml" --epoch 100 --batch-size 128 --cfg "models/yolov5s.yaml" --weights yolov5s.pt --name foodYolov5s --device 0,1


========================================================================================================================
validation

python val.py --data "../NIA_20220820_processed/data.yaml" --weights "runs/train/NIA_220820_Yolov5s5/weights/best.pt" --task "test" --accuracy


========================================================================================================================
detect

yolov5s
python detect.py --weights "runs/train/F5000_Yolov5s2/weights/best.pt" --source "../Foods5K/test/images"

yolov5m
python detect.py --weights "runs/train/F5K_Yolov5m10/weights/best.pt" --source "../Foods5K/test/images"

yolov5l
python detect.py --weights "runs/train/F5K_Yolov5l2/weights/best.pt" --source "../Foods20K/test/images"

yolov5x
python detect.py --weights "runs/train/F5K_Yolov5x2/weights/best.pt" --source "../Foods20K/test/images"

========================================================================================================================
micel
python -m torch.distributed.run --nproc_per_node 2 train.py --data "data/food/data.yaml" --epoch 100 --batch-size 128 --cfg "models/yolov5s.yaml" --weights yolov5s.pt --name foodYolov5s

python -m torch.distributed.run --nproc_per_node 2 train.py --data "../NIA_20220820_processed/data.yaml" --epoch 100 --batch-size 32 --cfg "models/yolov5s.yaml" --weights yolov5s.pt --name NIA_220820_Yolov5s --device 0,1

========================================================================================================================
2022.8.23 refactoring.

1. add multi processing code for build_dataset_from_spec()

2. R&D of saving console log as file
-> Done
3. Validation of test set
-> Done
========================================================================================================================
food-101 dataset

The Food-101 dataset consists of 101 food categories with 750 training and 250 test images per category, making a total of 101k images.
The labels for the test images have been manually cleaned, while the training set contains some noise.

========================================================================================================================
training tips.

https://docs.ultralytics.com/tutorials/training-tips-best-results/

========================================================================================================================
load instance and detect
--weights "runs/train/NIA_221014_Yolov5l2/weights/best.pt" --source "../NIA_20221014_processed/detect_test/"

========================================================================================================================
web_yolov5.py parameters

pycharm start
--host=0.0.0.0 --port=5000 --debug=True --use_reloader=False --data_folder=/home2/channelbiome/DataHub/WebData --weights=/home2/channelbiome/DataHub/yolov5/runs/train/NIA_221014_Yolov5l2/weights/best.pt

console start
python web_yolov5.py --host=0.0.0.0 --port=5000 --debug=True --use_reloader=False --device=cuda:0 --data_folder=/home2/channelbiome/DataHub/WebData --weights=/home2/channelbiome/DataHub/yolov5/runs/train/NIA_221014_Yolov5l2/weights/best.pt

========================================================================================================================
nia 2022 train................
--data "../NIA_2022_Dec_processed/data.yaml" --epoch 1 --batch-size 4 --cfg "models/yolov5s.yaml" --weights yolov5s.pt --name NIA_2022_Dec_Yolov5s

-m torch.distributed.run --nproc_per_node 2 train.py --data "../NIA_2022_Dec_processed/data.yaml" --epoch 1 --batch-size 4 --workers 16 --cfg "models/yolov5s.yaml" --weights yolov5s.pt --name NIA_2022_Dec_Yolov5s --device 0,1

-m torch.distributed.run --nproc_per_node 2 train.py --data "../NIA_2022_Dec_processed/data.yaml" --epoch 100 --batch-size 32 --workers 16 --cfg "models/yolov5l.yaml" --weights yolov5l.pt --name NIA_2022_Dec_Yolov5l --device 0,1

========================================================================================================================
nia 2022 continuous train.

* Resume
-m torch.distributed.run --nproc_per_node 2 train.py --resume "runs/train/NIA_2022_Dec_Yolov5s14/weights/last.pt"

*Start from pretrained
-m torch.distributed.run --nproc_per_node 2 train.py --data "../NIA_2022_Dec_processed/data.yaml" --epoch 1
--batch-size 24 --workers 16 --cfg "models/yolov5s.yaml" --weights "runs/train/NIA_2022_Dec_Yolov5s14/weights/last.pt"
--name NIA_2022_Dec_Yolov5s --device 0,1

========================================================================================================================
nia 2022 validation
python val.py --data "../NIA_2022_Dec_processed/data.yaml" --weights "runs/train/NIA_2022_Dec_Yolov5s14/weights/best.pt" --task "test" --workers 16
python val.py --data "../NIA_2022_Final_processed/data.yaml" --weights "runs/train/NIA_2022_Final_Yolov5l7/weights/best.pt" --task "test" --workers 32


========================================================================================================================
nia 2022 final

train
--data "../NIA_2022_Final_processed/data.yaml" --epoch 200 --batch-size 32 --workers 16 --cfg "models/yolov5l.yaml" --weights yolov5l.pt --name NIA_2022_Final_Yolov5l --device 0,1

========================================================================================================================
nia 2022 docker validation

python val.py --data "../nia_2022/data.yaml" --weights "../nia_2022/result/NIA_2022_Final_Yolov5l7/weights/best.pt" --task "test" --workers 32
python val.py --data "../nia_2022/data_docker.yaml" --weights "../nia_2022/result/NIA_2022_Final_Yolov5l7/weights/best.pt" --task "test" --workers 32

python val.py --data "../nia2022/data_docker.yaml" --weights "../nia2022/weights/best.pt" --task "test" --workers 8 --device cpu
python val.py --data "../nia_2022/data_docker.yaml" --weights "../nia_2022/weights/best.pt" --task "test" --workers 8 --device cpu

========================================================================================================================
aws p4
python -m torch.distributed.run --nproc_per_node 8 train.py --data "/usr/src/nia2022/data.yaml" --epoch 1 --batch-size 320 --workers 96 --cfg "models/yolov5l.yaml" --weights yolov5l.pt --name nia2022_Yolov5l --device 0,1,2,3,4,5,6,7