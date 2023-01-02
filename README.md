채널헬스케어 음식이미지 구분 모델 소스코드
======

원본 소스에 대한 GigHub url:
https://github.com/ultralytics/yolov5

#### 이 프로그램은 음식이미지 구분을 위한 학습 및 평가 도구로 yolov5 원본을 수정하여 작성하였음을 알립니다.

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
### - 음식구분 알고리즘은 yolov5중 large model인 yolov5l 모델을 바탕으로 진행
### - 수행 계획서의 성능 목표 mAP@0.5 80.0%를 상회한 mAP@0.5 86.5%를 달성. 

## License
[GPL-3.0 Lisence](https://github.com/ultralytics/yolov5/blob/master/LICENSE)