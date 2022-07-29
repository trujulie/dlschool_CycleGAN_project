# CycleGAN. PyTorch implementation

Реализация на PyTorch модели [CycleGAN](https://junyanz.github.io/CycleGAN/) для переноса стиля между двумя наборами изображений. Отличительной особенностью этой модели является то, что она не требует спаренных изображений для обучения.   

Например, для того, чтобы научить модель преобразовывать зимний пейзаж в летний не обязательно иметь фотографии одного и того же места с одного и того же ракурса в летний и зимний периоды. Достаточно собрать два датасета: любые фотографии пейзажей, сделанные в летний период, и любые фотографии зимних пейзажей. Модель сама научится строить отображение из одного пространства в другое. Примеры из оригинальной работы можно посмотреть [здесь](https://taesung.me/cyclegan/2017/03/25/yosemite-supplemental-best.html).


## Примеры генерации
Примеры того, как можно преобразовать картины Моне в фотографии и обратно:
![Alt-текст](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/best_examples/m2p.jpg)

Примеры того, как можно преобразовать обычные пейзажи в космические:
![Alt-текст](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/best_examples/l2s-2.jpg)
Первая строка - оригинальные изображения из датасета, вторая строка - сгенерированные изображения (с перенесенным стилем), третья строка - изображения, восстановленные из сгенерированных.


# Структура проекта

## `modules`
Здесь лежит весь рабочий код.


## `datasets`
Здесь должны лежать датасеты для обучения и тестирования модели.  
На данный момент доступны два набора данных (их можно скачать по ссылкам, кликнув на название): 
 - [landscape2space](https://drive.google.com/drive/folders/1CtHyhVe15RUVcbEtko8jpvajzkeIvBwK)    
   Фотографии пейзажей взяты из этого датасета [Landscape Pictures](https://www.kaggle.com/datasets/arnaud58/landscape-pictures) с платформы Kaggle.  
   Космические фотографии получены путем фильтрации датасета [Cosmos Images](https://www.kaggle.com/datasets/kimbosoek/cosmos-images) с платформы Kaggle.   
 - [monet2photo](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/monet2photo.zip).  
    
Каждый набор состоит из двух датасетов, относящихся к разным классам, которые разбиты на обучающую и тестовую подвыборки в одинаковом соотношении.

Рассмотрим, например, набор данных monet2photo. Он имеет следующую структуру каталога:
```
|-monet2photo
   |-testA
   |-testB
   |-trainA
   |-trainB
```
- trainA и testA - изображения картин Моне
- trainB и testB - фотографии пейзажей.


При желании можно создать свой набор данных. Для этого необходимо собрать два датасета, относящихся к разным классам, а затем разбить их на обучающую и тренировочную части по аналогии со структурой каталога monet2photo, которая описана выше.   
**Важно**:  название поддиректорий testA, testB, trainA, trainB изменять нельзя.


## `checkpoints`
Веса моделей, полученные в процессе обучения, будут сохраняться **здесь**.

На данный момент доступны предобученные веса для двух датасетов: [landscape2space](https://drive.google.com/drive/folders/1QsmJbrNlSFKz8SGzDVgOEiGWh-tvqD1c) и [monet2photo](https://drive.google.com/drive/folders/1QsmJbrNlSFKz8SGzDVgOEiGWh-tvqD1c). Их уже можно скачать по ссылкам, кликнув на название, и использовать для генерации собственных картинок :wink:.    


## `loss_history`
Значения функций потерь, полученные во время обучения или тестирования модели, будут храниться **здесь** (например, см. `loss_history/landscape2space train` или `loss_history/landscape2space test`, соответственно).


## `result_imgs`
Результаты генерации изображений в процессе обучения, тестирования или генерации будут храниться **здесь**.

Как будет выглядеть структура каталога в процессе обучения, тренировки и генерации, можно посмотреть на примере `landscape2space train`, `landscape2space test`, `landscape2space eval`.

## `best_examples`
Здесь можно найти несколько примеров удачной генерации изображений для наборов данных landscape2space и monet2photo.


## `requirements.txt`
Список необходимых пакетов и их версий, которые были использованы в процессе разработки проекта, можно найти в файле `requirements.txt`.


# Обучение

Чтобы начать обучение модели CycleGAN, нужно запустить скрипт [`train_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/train_model.py). 

```%bash
usage: train_model.py [-h] [--home_dir HOME_DIR] [--dataset_name DATASET_NAME] [--unaligned UNALIGNED]
                      [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] [--n_channels_input N_CHANNELS_INPUT]
                      [--n_channels_output N_CHANNELS_OUTPUT] [--num_epochs NUM_EPOCHS] [--start_epoch START_EPOCH]
                      [--decay_epoch DECAY_EPOCH] [--batch_size BATCH_SIZE]
                      [--checkpoint_interval CHECKPOINT_INTERVAL] [--device DEVICE] [--gpu_ids GPU_IDS]
                      [--n_cpu N_CPU] [--n_res_blocks N_RES_BLOCKS] [--lambda_cycle LAMBDA_CYCLE]
                      [--use_idt_loss USE_IDT_LOSS] [--lambda_idt LAMBDA_IDT] [--init_weight INIT_WEIGHT]
                      [--lr_discr LR_DISCR] [--lr_gen LR_GEN] [--b1 B1] [--b2 B2] [--n_photos N_PHOTOS]
                      [--pool_size POOL_SIZE] [--checkpoints_dir CHECKPOINTS_DIR] [--losshistory_dir LOSSHISTORY_DIR]
                      [--results_dir RESULTS_DIR] [--pretrained_weights_dir PRETRAINED_WEIGHTS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --home_dir HOME_DIR   path to folder with image datasets. default : ./datasets
  --dataset_name DATASET_NAME
                        name of the dataset. default : monet2photo
  --unaligned UNALIGNED
                        is dataset unaligned. default : True
  --img_height IMG_HEIGHT
                        crop image to this height. default : 256
  --img_width IMG_WIDTH
                        crop image to this width. default : 256
  --n_channels_input N_CHANNELS_INPUT
                        input image channels number. default: 3
  --n_channels_output N_CHANNELS_OUTPUT
                        output image channels number. default: 3
  --num_epochs NUM_EPOCHS
                        number of epochs to train. default: 200
  --start_epoch START_EPOCH
                        epoch number from which to start training. start_epoch should be divisible by
                        checkpoint_interval. default: 0
  --decay_epoch DECAY_EPOCH
                        number of epoch from which to start lr decay. default: 100
  --batch_size BATCH_SIZE
                        size of batch. default: 1
  --checkpoint_interval CHECKPOINT_INTERVAL
                        how often to save checkpoints. default: 20
  --device DEVICE       device type. default : cuda
  --gpu_ids GPU_IDS     IDs of gpus to use: e.g. 0 0,1,2, 0,2. default : 0
  --n_cpu N_CPU         number of cpus to use. default : 10
  --n_res_blocks N_RES_BLOCKS
                        number of residual blocks in generator. default: 9
  --lambda_cycle LAMBDA_CYCLE
                        cycle loss weight. default: 10
  --use_idt_loss USE_IDT_LOSS
                        whether to use identity loss. default: True
  --lambda_idt LAMBDA_IDT
                        identity loss weight. default: 5
  --init_weight INIT_WEIGHT
                        model weights will be initialized with normal distribution with mean=0 and std=init_weight.
                        default: 0.02
  --lr_discr LR_DISCR   discriminator optimizer learning rate. default: 2e-4
  --lr_gen LR_GEN       generator optimizer learning rate. default: 2e-4
  --b1 B1               Adam optimizer beta1 coefficient. default: 0.5
  --b2 B2               Adam optimizer beta2 coefficient. default: 0.999
  --n_photos N_PHOTOS   number of images to check results on. default: 5
  --pool_size POOL_SIZE
                        size of image pool to store previously generated images. default: 50
  --checkpoints_dir CHECKPOINTS_DIR
                        directory to store checkpoints. if not specified, it is set to "(dataset_name) train (year-
                        month-day Hour-Min)"
  --losshistory_dir LOSSHISTORY_DIR
                        directory to store loss history. if not specified, it is set to "(dataset_name) train (year-
                        month-day Hour-Min)"
  --results_dir RESULTS_DIR
                        folder to save result images. if not specified, it is set to "(dataset_name) train (year-
                        month-day Hour-Min)"
  --pretrained_weights_dir PRETRAINED_WEIGHTS_DIR
                        directory where pretrained weights are saved
```

В директории `checkpoints` создается папка --checkpoints_dir. Каждые checkpoint_interval эпох в папку `epoch_i` будут сохраняться выученные за i эпох веса моделей. 
**Стоит обратить внимание**, что i начинается с 0, поэтому крайние обученные веса моделей будут лежать в папке с именем `epoch_(num_epochs-1)`.

В директории `loss_history` создается папка --losshistory_dir. Каждые checkpoint_interval эпох в файл `epoch_i.csv` будут сохраняться средние значения лосс функций по каждому батчу, полученные за i эпох. В конце обучения в файл `train_loss.csv` будут сохранены средние значения лосс функций на каждой эпохе.

Сгенерированные картинки сохраняются в поддиректорию --results_dir папки `result_imgs`. В --results_dir будут созданы 2 папки:
 - В папке AtoB сохраняются результаты генерации изображений из класса А в класс В,
 - В папке ВtoА сохраняются результаты генерации изображений из класса В в класс А.   
 
Каждые checkpoint_interval эпох в эти папки в файлы `epoch_i.jpg` будут сохраняться результаты генерации для n_photos изображений тестового множества. Для наглядности на каждой эпохе берутся одни и те же исходные изображения.   
Сгенерированные изображения - это сетка из 3 х n_photos картинок. Первая строка - исходные изображения (например, A), вторая - сгенерированные (соответственно, B), третья - изображения, восстановленные из сгенерированных (соответственно, A).

**Стоит обратить внимание**, что i начинается с 0, поэтому крайние результаты генерации будут лежать в файле с именем `epoch_(num_epochs-1).jpg`
 
Если название какой-либо из директорий --checkpoints_dir, --results_dir или --losshistory_dir не указано, берется дефолтное название, которое выглядит следующим образом: `dataset_name test yy-mm-dd hh-mm`, где yy-mm-dd hh-mm - текущие год-число-месяц часы-минуты.

Примеры запуска: 
 - **monet2photo**  
```bash
python train_model.py --dataset_name=monet2photo 
```
- **landscape2space**  
```bash
python train_model.py --dataset_name=landscape2space --use_idt_loss=False
```

# Тестирование

Чтобы протестировать качество работы модели CycleGAN, нужно запустить скрипт [`test_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/test_model.py).

```%bash
usage: test_model.py [-h] [--home_dir HOME_DIR] [--dataset_name DATASET_NAME] [--unaligned UNALIGNED]
                     [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH] [--n_channels_input N_CHANNELS_INPUT]
                     [--n_channels_output N_CHANNELS_OUTPUT] [--batch_size BATCH_SIZE] [--device DEVICE]
                     [--gpu_ids GPU_IDS] [--n_cpu N_CPU] [--n_res_blocks N_RES_BLOCKS] [--lambda_cycle LAMBDA_CYCLE]
                     [--use_idt_loss USE_IDT_LOSS] [--lambda_idt LAMBDA_IDT] [--init_weight INIT_WEIGHT]
                     [--pool_size POOL_SIZE] [--losshistory_dir LOSSHISTORY_DIR] [--results_dir RESULTS_DIR]
                     [--pretrained_weights_dir PRETRAINED_WEIGHTS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --home_dir HOME_DIR   path to folder with image datasets. default : ./datasets
  --dataset_name DATASET_NAME
                        name of the dataset. default : monet2photo
  --unaligned UNALIGNED
                        is dataset unaligned. default : True
  --img_height IMG_HEIGHT
                        crop image to this height. default : 256
  --img_width IMG_WIDTH
                       crop image to this width. default : 256
  --n_channels_input N_CHANNELS_INPUT
                        input image channels number. default: 3
  --n_channels_output N_CHANNELS_OUTPUT
                        output image channels number. default: 3
  --batch_size BATCH_SIZE
                        size of batch. default: 1
  --device DEVICE       device type. default : cuda
  --gpu_ids GPU_IDS     IDs of gpus to use: e.g. 0 0,1,2, 0,2. default : 0
  --n_cpu N_CPU         number of cpus to use. default : 10
  --n_res_blocks N_RES_BLOCKS
                        number of residual blocks in generator. should be the same as in pretrained model. default: 9
  --lambda_cycle LAMBDA_CYCLE
                        cycle loss weight. default: 10
  --use_idt_loss USE_IDT_LOSS
                        whether to use identity loss. default: True
  --lambda_idt LAMBDA_IDT
                        identity loss weight. default: 5
  --init_weight INIT_WEIGHT
                        model weights will be initialized with normal distribution with mean=0 and std=init_weight.
                        default: 0.02
  --pool_size POOL_SIZE
                        size of image pool to store previously generated images. default: 50
  --losshistory_dir LOSSHISTORY_DIR
                        directory to store loss history. if not specified, it is set to "(dataset_name) test (year-
                        month-day Hour-Min)"
  --results_dir RESULTS_DIR
                        folder to save result images. if not specified, it is set to "(dataset_name) test (year-month-
                        day Hour-Min)"
  --pretrained_weights_dir PRETRAINED_WEIGHTS_DIR
                        directory where pretrained weights for all models are saved
```

В качестве аргументов обязательно нужно указать аргумент --pretrained_weights_dir путь в директорию, в которой хранятся предобученные веса моделей.

В директории `loss_history` создается папка --losshistory_dir. Значения лоссов на каждой батче будут сохраняться в эту директорию в файле `test_loss.csv`.

Сгенерированные картинки сохраняются в поддиректорию --results_dir папки `result_imgs`. В --results_dir будут созданы 2 папки:
 - В папке AtoB сохраняются результаты генерации изображений из класса А в класс В,
 - В папке ВtoА сохраняются результаты генерации изображений из класса В в класс А.
Сгенерированные изображения - это сетка из 3хbatch_size картинок. Первая строка - исходные изображения (например, A), вторая - сгенерированные (соответственно, B), третья - изображения, восстановленные из сгенерированных (соответственно, A).
 
Если название какой-либо из директорий --results_dir или --losshistory_dir не указано, берется дефолтное название, которое выглядит следующим образом: `dataset_name test yy-mm-dd hh-mm`, где yy-mm-dd hh-mm - текущие год-число-месяц часы-минуты.

Примеры запуска: 
 - **monet2photo**  
```bash
python test_model.py --dataset_name=monet2photo --pretrained_weights_dir=./checkpoints/monet2photo/ 
```
- **landscape2space**  
```bash
python test_model.py --dataset_name=landscape2space --pretrained_weights_dir=./checkpoints/landscape2space/ 
```


# Генерация

Чтобы сгенерировать результат для своего изображения, нужно запустить скрипт [`eval_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/eval_model.py).   

```%bash
usage: eval_model.py [-h] [--path_file PATH_FILE] [--img_height IMG_HEIGHT] [--img_width IMG_WIDTH]
                     [--n_channels_input N_CHANNELS_INPUT] [--n_channels_output N_CHANNELS_OUTPUT]
                     [--n_res_blocks N_RES_BLOCKS] [--device DEVICE] [--path_checkpoints PATH_CHECKPOINTS]
                     [--results_dir RESULTS_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --path_file PATH_FILE
                        path to real image
  --img_height IMG_HEIGHT
                        crop image to this height. default : 256
  --img_width IMG_WIDTH
                        crop image to this width. default : 256
  --n_channels_input N_CHANNELS_INPUT
                        input image channels number. default: 3
  --n_channels_output N_CHANNELS_OUTPUT
                        output image channels number. default: 3
  --n_res_blocks N_RES_BLOCKS
                        number of residual blocks in generator. should be the same as in pretrained model. default: 9
  --device DEVICE       device type. default : cuda
  --path_checkpoints PATH_CHECKPOINTS
                        path to generator pretrained weights file
  --results_dir RESULTS_DIR
                        folder to save result images. if not specified it is set to "eval (year-month-day Hour-Min)"
```

Путь к выбранному изображению нужно передать через аргумент --path_file.    
В аргументе --path_checkpoints нужно указать путь к файлу, в котором лежат веса генератора для правильного класса. То есть, например, если мы выбрали изображение картины Моне (класс А в датасете), то и в аргумент  --path_checkpoints мы передаем веса генератора, обученного переводить изображения из класса A в класс B.   

В этом проекте веса моделей генерации всегда сохраняются с именами `gen_A_weights.pt` и `gen_B_weights.pt`. Чтобы преобразовать картинку из класса A в класс B, надо выбирать файл с именем `gen_A_weights.pt`, иначе - `gen_B_weights.pt`.

Результат работы сохраняется в поддиректорию --results_dir папки `result_imgs`. Если аргумент --results_dir не указан, по умолчанию будет сформирована директория с именем `eval yy-mm-dd hh-mm`, где yy-mm-dd hh-mm - текущие год-число-месяц часы-минуты.

Примеры запуска: 
 - **monet2photo**  
```bash
python eval_model.py --path_file=./datasets/monet2photo/testA/00350.jpg  --path_checkpoints=./checkpoints/monet2photo/gen_A_weights.pt
```
- **landscape2space**  
```bash
python eval_model.py --path_file=./datasets/landscape2space/testA/00000189_(4).jpg  --path_checkpoints=./checkpoints/landscape2space/gen_A_weights.pt
```

