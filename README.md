채널헬스케어 음식이미지 구분 모델 소스코드
======

원본 소스에 대한 GigHub url:
https://github.com/ultralytics/yolov5

#### 이 프로그램은 음식이미지 구분을 위한 학습 및 평가 도구로 yolov5 원본을 일부 수정하여 작성하였음을 알립니다.

## Install
파이썬 및 파이토치 필요 조건을 맞추고(Python >= 3.8.0 and PyTorch >= 1.7) 아래의 명령을 실행한다.
```
git clone https://github.com/channel-healthcare/Food_Detector #clone
cd yolov5
pip install -r requirements.txt
```
## Train
### - single GPU
```
python train.py --data [dataset path]/data.yaml 
                --epoch 200
                --cfg "models/yolov5l.yaml"
                --weights yolov5l.pt
                --batch-size 24
```
### - multi GPU
#### * epoch는 200 이내에서 학습이 완료되었고, 
#### * batch-size는 v100 / 2-GPU에서 24로 수행

```
python -m torch.distributed.run --nproc_per_node [gpu count] train.py 
                                --data "[dataset path]/data.yaml" 
                                --epoch 200 
                                --cfg "models/yolov5l.yaml"
                                --weights yolov5l.pt
                                --batch-size 320 
                                --workers [cpu core count]  
                                --device [gpu index ex: 0, 1, 2, 3, ...]
```
## Validation
```buildoutcfg
python val.py --data "[dataset path]/data.yaml"
              --weights "[train result path]/weights/best.pt" 
              --task "test" 
              --workers [cpu core count]
```

## 모델 정보
### 모델 description
YOLOv5는 object detection 분야에서 가장 많이 이용되고 있는 모델로 trade-off 속성을 가진 성능과 속도를 합리적인 수준에서 만족시킬 수 있는 모델이다. 이번 과제에서는 속도보다는 성능을 고려하여 YOLOv5l 모델을 이용하였다.
### 모델 아케텍쳐
![Architecture](https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png)
### Input
shape = [b, c, w, h], where b = batch-size, c = channel, w =width, h = height  
### Output
shape = [[P3, P4, P5], 1, Detect, [nc, anchors]], where P3, P4, P5 is header's layer output, nc = class number, anchors = anchor boxes 
### Task
Object Detection
### Training dataset
음식 이미지 204종 \
Total image count = 537,209\
train : valid : test = 8 : 1 : 1 = 429,687 : 53,718 : 53,804

### Training 요소들
####Loss function
YOLOv5는 3가지 손실함수의 조합으로 되어 있다.\
- Classes loss(BCE loss) \
- Objectness loss(BCE loss) \
- Location loss(CIoU loss)\
![Loss Function](https://camo.githubusercontent.com/af2d80e8094c28221f1d2b7bdf11e231c5927102c3323dd2c572cb2561c51aeb/68747470733a2f2f6c617465782e636f6465636f67732e636f6d2f7376672e696d6167653f4c6f73733d2535436c616d6264615f314c5f253742636c732537442b2535436c616d6264615f324c5f2537426f626a2537442b2535436c616d6264615f334c5f2537426c6f63253744) \
####Optimizer
SDG(Stochastic Gradient Descent) - default optimizer
####Hyperparamereter
epoch = 100 (53 early stopped)\
batch-size = 24 \
leaning rate ![learning rate](D:\Development\1.Python\Food_Detector\data\images\lr01.png)
 
### Evaluation metric
#### - 수행 계획서의 성능 목표 mAP@0.5 80.0%를 상회한 mAP@0.5 86.5%를 달성. 
mAP@0.5
![map50](D:\Development\1.Python\Food_Detector\data\images\mAP50.png)
precision
![precision](D:\Development\1.Python\Food_Detector\data\images\precision.png)
recall
![recall](D:\Development\1.Python\Food_Detector\data\images\recall.png)
## License
[GPL-3.0 Lisence](https://github.com/ultralytics/yolov5/blob/master/LICENSE)