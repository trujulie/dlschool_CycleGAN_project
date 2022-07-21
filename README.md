# CycleGAN. PyTorch implementation

Реализация на PyTorch модели [CycleGAN](https://junyanz.github.io/CycleGAN/) для переноса стиля между двумя наборами изображений.

Отличительной особенностью этой модели является то, что она не требует спаренных изображений для обучения. Например, для того, чтобы научить модель преобразовывать зимний пейзаж в летний не обязательно иметь фотографии одного и того же места с одного и того же ракурса в летний и зимний периоды. Достаточно собрать два датасета: любые фотографии пейзажей, сделанные в летний период, и любые фотографии зимних пейзажей. Модель сама научится строить отображение из одного пространства в другое. Примеры можно посмотреть [здесь](https://taesung.me/cyclegan/2017/03/25/yosemite-supplemental-best.html).

Список необходимых пакетов можно найти в файле `requirements.txt'.

# Обучение

Чтобы начать обучение модели CycleGAN, необходимо запустить скрипт [`train_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/train_model.py). 

# Тестирование

Чтобы протестировать качество работы модели CycleGAN, необходимо запустить скрипт [`test_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/test_model.py).

# Генерация

Чтобы сгенерировать результат для своего изображения, необходимо запустить скрипт [`eval_model.py`](https://github.com/trujulie/dlschool_CycleGAN_project/blob/main/eval_model.py).
