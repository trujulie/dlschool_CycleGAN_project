# CycleGAN. PyTorch implementation

Реализация на PyTorch модели [CycleGAN](https://junyanz.github.io/CycleGAN/) для переноса стиля между двумя наборами изображений.

Отличительной особенностью этой модели является то, что она не требует спаренных изображений для обучения. Например, для того, чтобы научить модель преобразовывать зимний пейзаж в летний не обязательно иметь фотографии одного и того же места с одного и того же ракурса в летний и зимний периоды. Достаточно собрать два датасета: любые фотографии пейзажей, сделанные в летний период, и любые фотографии зимних пейзажей. Модель сама научится строить отображение из одного пространства в другое. Примеры можно посмотреть [здесь](https://taesung.me/cyclegan/2017/03/25/yosemite-supplemental-best.html).

Пример того, как можно преобразовать картины Моне в фотографии и обратно:
![Alt-текст](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/best_examples/A_to_B_2022-07-02_09-53.png)


# Структура проекта

## `modules`
Здесь лежит весь рабочий код.


## `datasets`
В директории `datasets` хранятся датасеты для обучения и тестирования модели.  
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


При желании можно создать свой набор данных. Для этого необходимо собрать два датасета, относящихся к разным классам, а затем разбить их на обучающую и тренировочную части по аналогии со структурой каталога monet2photo, которая описана выше. 
**Важно**:  название поддиректорий testA, testB, trainA, trainB изменять нельзя.


## `checkpoints`
Веса моделей, полученные в процессе обучения, будут сохраняться **здесь**.

На данный момент там лежат две поддиректории: landscape2space и monet2photo. В них хранятся предобученные веса моделей для этих двух датасетов. Их уже можно использовать для генерации собственных картинок :wink:.

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

