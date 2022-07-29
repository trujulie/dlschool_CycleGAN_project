# `datasets`
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

