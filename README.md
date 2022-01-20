# Phoneme Boundary Detection using Learnable Segmental Features (ICASSP 2020) + Ensemble model

SegFeat(Phoneme Boundary Detection using Learnable Segmental Features) 모델과 Ensemble model을 각각 훈련 시킬 수 있는 모델이다.

<img src="https://user-images.githubusercontent.com/77380115/150287333-0040dd97-aab2-4b58-b740-c74a97841898.PNG"  width="800" height="270"/>
트레이닝 순서는 MFA, SegFeat, Ensemble model 순서로 진행해야 한다.

MFA의 트레이닝 방법은 2번 항목

SegFeat는 3번 항목

Ensemble model은 4번 항목에 적혀 있다.


## Dependencies

```
python==3.9
torch==1.10.1
librosa==0.8.1
textgrid
loguru==0.4.1
boltons==20.0.0
pandas==1.3.5
pytorch-lightning==1
SoundFile==0.10.3.post1
test-tube==0.7.5
torchaudio
torchvision
tqdm==4.42.1
g2p_en
Levenshtein
argparse
```

## Usage

### 1. Clone the repository
```
git clone https://open.oss.navercorp.com/NES/Segmentation_Ensemble_Model.git
cd Segmentation_Ensemble_Model
```

### 2. Data structure
'dataloader.py'에서 경로가 지정되어 있으며 구조는 아래와 같다.
```
timit_directory
│
└───test
│   │   X.wav
│   └─  X.phn
|   └─  X.TextGrid
│   └─  X.lab
│
└───train
    │   Y.wav
    └─  Y.phn
    └─  Y.TextGrid
    └─  Y.lab
```

#### Audio
```
sampling_rate : 16000
sample type : 1-channel pcm
max_length : 5s
```
#### phn
```
0 9640 h#
9640 11240 sh
11240 12783 iy
12783 14078 hv
14078 16157 ae
16157 16880 dcl
...
```
wav file의 음소와 onset, offset이 적혀져 있는 파일이다.
(onset, offset, 음소)로 구성되어 있다.
SegFeat와 Ensemble model의 지도학습 용도와 성능 평가 용도로 쓰인다.


#### lab 
```
She had your dark suit in greasy wash water all year
```
wav file의 텍스트 레이블이다.
MFA에서 TextGrid를 뽑을 때 사용된다.


#### TextGrid 
```
    item [1]:
        class = "IntervalTier" 
        name = "words" 
        xmin = 0 
        xmax = 1 
        intervals: size = 13 
        intervals [1]:
            xmin = 0 
            xmax = 0.11 
            text = "" 
        intervals [2]:
            xmin = 0.11 
            xmax = 0.38 
            text = "she" 
    item [2]:
        class = "IntervalTier" 
        name = "phones" 
        xmin = 0 
        xmax = 1 
        intervals: size = 36 
        intervals [1]:
            xmin = 0 
            xmax = 0.11 
            text = "" 
        intervals [2]:
            xmin = 0.11 
            xmax = 0.28 
            text = "SH" 
        intervals [3]:
            xmin = 0.28 
            xmax = 0.38 
            text = "IY1" 
```
MFA의 결과로 나오는 값이며, xmin(onset), xmax(offset), text(phoneme or words)로 구성되어 있다.
Ensemble model의 input으로 들어간다.

#### Timit data 받는 법
```
mkdir timit
cd timit
wget https://data.deepai.org/timit.zip
unzip timit.zip
cd ../
python timit.py
rm -r timit
```

#### MFA training

아래와 같은 과정을 통해 TextGrid를 뽑을 수 있다.
```
conda create -n aligner -c conda-forge montreal-forced-aligner openblas python=3.8 openfst pynini ngram baumwelch sox
conda activate aligner
wget http://www.openslr.org/resources/11/librispeech-lexicon.txt
mfa train ./timit_directory/train librispeech-lexicon.txt ./out -o timit
mv ./out/* ./timit_directory/train
mfa align ./timit_directory/test librispeech-lexicon.txt timit.zip ./out
mv ./out/* ./timit_directory/test
```
전체 훈련 시간은 1시간 정도 소요된다. 

서버에는 '/home1/irteam/data' 디렉토리 안에 'timit_g2p'(g2p로 진행한 MFA)와 'timit_gold'(정답 음소로 진행한 MFA)가 각각 있다.


### 3. SegFeat_Training and Test

#### Training
```
python main.py --wav_path (timit directory) --run_dir (tensorboard 및 ckpt 저장되는 곳) --gpus (사용할 gpu 번호)
```
* run_dir/ckpt에 1epoch 마다 SegFeat model이 저장된다.
* 논문에서는 150 epoch을 진행하였으며, 1 epoch당 1시간 정도 소요된다. 
* --load_ckpt를 통해서 저장된 모델을 불러올 수 있다.


#### Test
```
python main.py --wav_path (timit directory) --run_dir (tensorboard 및 ckpt 저장되는 곳) --gpus (사용할 gpu 번호) --load_ckpt (SegFeat 모델 체크포인트) --test
```
* 뒤에 test와 pretrain model을 load 해주는 것으로 테스트 가능하다.
* 결과로는 테스트 셋의 precision-recall, f1-score, Accuracy를 얻을 수 있다.


### 4. Ensemble_Training and Test

#### Training
```
python main.py --wav_path (timit directory) --run_dir (tensorboard 및 ckpt 저장되는 곳) --gpus (사용할 gpu 번호) --ans
```
* --ans의 플래그를 세우는 것으로 Ensemble model의 training이 가능하다.
* run_dir/ckpt에는 Segfeat 체크포인트가 아닌 Ensemble model의 체크포인트가 저장된다.
* 20 epoch train 진행했으며, 1epoch당 1시간 정도 소요된다.

#### Test
```
python main.py --wav_path (timit directory) --run_dir (tensorboard 및 ckpt 저장되는 곳) --gpus (사용할 gpu 번호) --load_ckpt (SegFeat 모델 체크포인트) --load_s (Ensemble 모델 체크포인트) --test --ans
```
* --ans 플래그와 SegFeat pretrain(--load_ckpt), Ensemble pretrain(--load_s)를 넣고 --test 플래그를 세우는 것으로 진행할 수 있다.
* 결과로는 precision-recall, f1-score, accuracy, PER을 얻을 수 있다.
* --output_save라는 플래그를 세우면, infer에 test한 결과들이 저장되며,
```
python evalu.py --path ./infer
```
를 통해서 MFA, SegFeat, Ensemble model 각각의 PER(Subsititution, Deletion, Insertion), Precision-recall, f1-score, R-value, Accuracy를 볼 수 있다.


### 5. Pretrained Model, Output

바로 Train하고 Test를 진행하기 위해, Pretrained Model과 Output을 미리 뽑아 올려두었다.

* SegFeat Pretrained model : SegFeat_pretrain.ckpt
* Ensemble Pretrained model(g2p) : ans_pretrain_g2p.ckpt
* Ensemble Pretrained model(gold) : ans_pretrain_gold.ckpt
* Ensemble model Output(g2p) : infer_g2p
* Ensemble model Output(gold) : infer_gold
* g2p는 MFA training 할 때 g2p로 트레이닝한 모델이다.
* gold는 MFA training 할 때 정답 음소로 트레이닝한 모델이다.
* output은 'evalu.py'를 통해서 성능을 알아볼 수 있다.
```
python evalu.py --path ./infer_g2p
```

### TensorBoard
```
tensorboard ./run_dir/segmentation_experiment/lightning_logs/
```
