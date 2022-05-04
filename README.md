# Задание №2
### Курс «Вариационные методы обработки изображений».

Программа реализует метод сегментации изображения с помощью активных контуров.

Программа поддерживает запуск из командной строки со строго определённым форматом команд:
%programname% (input_image) (initial_snake) (output_image) (alpha) (beta) (tau) (w_line) (w_edge) (kappa)
Аргументы:
input_image   — входное изображение (имя файла)
initial_snake — начальное приближение для активного контура (имя файла)
output_image  — выходное изображение (имя файла)
alpha	      —	параметр внутренней энергии, отвечающий за растяжимость контура
beta       	  — параметр внутренней энергии, отвечающий за жесткость контура
tau	 	      — шаг градиентного спуска
w_line	      — вес слагаемого интенсивности во внешней энергии
w_edge	      — вес слагаемого границ во внешней энергии
kappa	      — вес balloon force

В test_data находятся исходные изображения, начальные приближения для активных контуров, маска сегментации ground truth, а также результаты работы алгоритма.

### Параметры для изображений:
| input_image    | alpha | beta | tau | w_line | w_edge | kappa | **IoU** |
| -------------- | ----- | ---- | --- | ------ | ------ | ----- | ------- |
| astranaut.png  | 0.7   | 0.1  | 11  |  0.1   | 0.9    | 0.0001| 98.36   |
| coffee.png     | 0.7   | 0.1  | 11  | -0.1   | 1.9    | 0.0001| 99.05   |
| coins.png      | 2.1   | 0.1  | 15  | -0.1   | 2.5    | 0.002 | 95.00   |
| microarray.png | 1.0   | 0.1  | 11  | -0.1   | 1.9    | 0.03  | 97.59   |
| nucleus.png    | 1.0   | 0.1  | 11  |  0.1   | 1.9    | 0.002 | 97.33   |