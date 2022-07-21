# CycleGAN. PyTorch implementation

Реализация на PyTorch модели [CycleGAN](https://junyanz.github.io/CycleGAN/) для переноса стиля между двумя наборами изображений.

Отличительной особенностью этой модели является то, что она не требует спаренных изображений для обучения. Например, для того, чтобы научить модель преобразовывать зимний пейзаж в летний не обязательно иметь фотографии одного и того же места с одного и того же ракурса в летний и зимний периоды. Достаточно собрать два датасета: любые фотографии пейзажей, сделанные в летний период, и любые фотографии зимних пейзажей. Модель сама научится строить отображение из одного пространства в другое. Примеры можно посмотреть [здесь](https://taesung.me/cyclegan/2017/03/25/yosemite-supplemental-best.html).


# Структура проекта

## `datasets`
В директории `datasets` хранятся датасеты для обучения и тестирования модели.
На данный момент там лежат две поддиректории: landscape2space и monet2photo.


## `checkpoints`
В директории `checkpoints` будут сохраняться веса моделей в процессе обучения.
На данный момент там содержится два набора данных: landscape2space и monet2photo. 
Каждый набор состоит из двух датасетов, относящихся к разным классам, которые разбиты на обучающую и тестовую подвыборки в одинаковом соотношении.

Рассмотрим, например, набор данных monet2photo. Он имеет следующую структуру каталога:
```
|-monet2photo
|--testA
|--testB
|--trainA
|--trainB
```
- trainA и testA - изображения картин Моне
- trainB и testB - фотографии пейзажей.

При желании можно создать свой набор данных. Для этого необходимо собрать два датасета, относящихся к разным классам, а затем разбить их на обучающую и тренировочную части по аналогии со структурой каталога monet2photo, которая описана выше. **Важно**:  название поддиректорий testA, testB, trainA, trainB изменять нельзя.


## `loss_history`

## `result_imgs`

## `best_examples`

## `modules`



## `requirements.txt`
Список необходимых пакетов можно найти в файле `requirements.txt'.

# Обучение

Чтобы начать обучение модели CycleGAN, необходимо запустить скрипт [`train_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/train_model.py). 

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

Чтобы протестировать качество работы модели CycleGAN, необходимо запустить скрипт [`test_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/test_model.py).

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

Чтобы сгенерировать результат для своего изображения, необходимо запустить скрипт [`eval_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/eval_model.py).

Примеры запуска: 
 - **monet2photo**  
```bash
python eval_model.py --path_file=./datasets/monet2photo/testA/00350.jpg  --path_checkpoints=./checkpoints/monet2photo/gen_A_weights.pt
```
- **landscape2space**  
```bash
python eval_model.py --path_file=./datasets/landscape2space/testA/00000189_(4).jpg  --path_checkpoints=./checkpoints/landscape2space/gen_A_weights.pt
```

