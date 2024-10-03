import cv2
import numpy as np
from PIL import Image

#Функция для удаления лишних внутренних ребер
def minus_red(img):
    index = 0
    for x in img:
        index_1 = 0
        for color in x:
            if color[0] == 255:
                img[index][index_1] = [255, 255, 255]
            index_1 += 1
        index += 1
#Функция для определения доминирующего значения цвета пикселя
def needed_color(arr):
    l_1 = []
    l_2 = []
    for element in arr:
        if element not in l_1 and element < 250:
            l_1.append(element)
            l_2.append(1)
        elif element in l_1:
            l_2[l_1.index(element)] += 1
    if len(l_2) == 0:
        return None  # Или другое значение по умолчанию
    return int(l_1[l_2.index(np.max(l_2))])

img = cv2.imread('words.png')
img_clear = np.zeros_like(img, dtype='uint8')
#Окрашивание фона в белый цвет
img_clear[:,:,0] = 255
img_clear[:,:,1] = 255
img_clear[:,:,2] = 255
### ПЕРВЫЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Нанесение черных квадратов на изображение)
print('Первый этап обработки...')
#Перевод изображения в черно-белый формат
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#Обнаружеине краев (внешних границ изображения)
gray = cv2.Canny(gray, 30, 30)
#Установка пороговых значений цвета пикселя
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
#Поиск контуров изображения
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img_1 = img.copy()

##Нанесение на изображение черных квадратов
for idx, contour in enumerate(contours):
    # Получение координат найденных контуров
    (x, y, w, h) = cv2.boundingRect(contour)
    if w < 41 and h >= 41:
        cv2.rectangle(img_1, (x - 6, y), (x + w + 6, y + h), (0, 0, 0), cv2.FILLED)
    elif w >= 41 and h < 41:
        cv2.rectangle(img_1, (x, y - 6), (x + w, y + h + 6), (0, 0, 0), cv2.FILLED)
    elif w < 41 and h < 41:
        cv2.rectangle(img_1, (x - 6, y - 6), (x + w + 6, y + h + 6), (0, 0, 0), cv2.FILLED)
    elif w >= 41 and h >= 41:
        cv2.rectangle(img_1, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)


### ВТОРОЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Нахождение контуров вокруг черных квадратов и скрытие найденных слов)
print('Второй этап обработки...')
# Перевод изображения в черно-белый формат
gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
#Применение размытия
gray_1 = cv2.GaussianBlur(gray_1, (13, 13), 0)
#Обнаружеине краев (внешних границ изображения)
gray_1 = cv2.Canny(gray_1, 80, 80)
#Установка пороговых значений цвета пикселя
ret_1, thresh_1 = cv2.threshold(gray_1, 200, 255, cv2.THRESH_BINARY)
#Поиск контуров изображения
contours_1, hierarchy_1 = cv2.findContours(thresh_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
output = img.copy()
img_2 = img.copy()


for idx_1, contour_1 in enumerate(contours_1):
    # Получение координат найденных контуров
    (x, y, w, h) = cv2.boundingRect(contour_1)
    if w * h > 2000 and w * h < 32000:
        if w != 184 and h != 172:
            cv2.rectangle(img_2, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)
            # Вырезание каждой буквы
            img_cropped = img[y+10:y + h-10, x+10:x + w-10]

            # Создание массивов из набора каждого цвета в контуре
            b_array = np.array(img_cropped[:, :, 0]).flatten()
            g_array = np.array(img_cropped[:, :, 1]).flatten()
            r_array = np.array(img_cropped[:, :, 2]).flatten()

            # Определение компонент доминирующего цвета в контуре
            average_blue = needed_color(b_array)
            average_green = needed_color(g_array)
            average_red = needed_color(r_array)

            # Рисование фона для каждой буквы на чистом холсте
            cv2.rectangle(img_clear, (x - 5, y - 5), (x + w + 5, y + h + 5),
                          (255 - average_blue, 255 - average_green, 255 - average_red), -1)





### ТРЕТИЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Нахождение контуров вокруг черных квадратов на изображении с оставшимися словами
# и скрытие найденных слов)
print('Третий этап обработки...')
#Обработка изображения и нахождение контуров
gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
gray = cv2.Canny(gray, 30, 30)
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
output_2 = img_2.copy()
img_3 = img_2.copy()
output_2_special = img_2.copy()

for idx, contour in enumerate(contours):
    # Получение координат найденных контуров
    (x, y, w, h) = cv2.boundingRect(contour)
    if w * h > 1500 and w * h < 32000:
        if w < 41 and h >= 41:
            cv2.rectangle(output_2, (x - 2, y), (x + w + 3, y + h), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(output_2_special, (x - 2, y), (x + w + 3, y + h), (255, 255, 255), cv2.FILLED)
        elif w >= 41 and h < 41:
            cv2.rectangle(output_2, (x, y - 2), (x + w, y + h + 3), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(output_2_special, (x - 2, y), (x + w + 3, y + h), (255, 255, 255), cv2.FILLED)
        elif w < 41 and h < 41:
            cv2.rectangle(output_2, (x - 2, y - 2), (x + w + 3, y + h + 3), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(output_2_special, (x - 2, y), (x + w + 3, y + h), (255, 255, 255), cv2.FILLED)
        elif w >= 41 and h >= 41:
            cv2.rectangle(output_2, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)
            cv2.rectangle(output_2_special, (x - 2, y), (x + w + 3, y + h), (255, 255, 255), cv2.FILLED)


### ЧЕТВЕРТЫЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Нахождение отдельных ненайденных букв и нанесение черных квадратов на изображение)
print('Четвертый этап обработки...')
#Обработка изображения и нахождение контуров
gray = cv2.cvtColor(output_2_special, cv2.COLOR_BGR2GRAY)
gray = cv2.Canny(gray, 30, 30)
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for idx, contour in enumerate(contours):
    # Получение координат найденных контуров
    (x, y, w, h) = cv2.boundingRect(contour)
    if w * h > 1200:
        if w < 34 and h >= 42:
            cv2.rectangle(output_2, (x - 4, y), (x + w + 4, y + h), (0, 0, 0), cv2.FILLED)
        elif w >= 34 and h < 42:
            cv2.rectangle(output_2, (x, y - 4), (x + w, y + h + 4), (0, 0, 0), cv2.FILLED)
        elif w < 34 and h < 42:
            cv2.rectangle(output_2, (x - 4, y - 4), (x + w + 4, y + h + 4), (0, 0, 0), cv2.FILLED)
        elif w >= 34 and h >= 42:
            cv2.rectangle(output_2, (x, y), (x + w, y + h), (0, 0, 0), cv2.FILLED)



### ПЯТЫЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Сборка контуров, найденных для частей слов и отдельных букв в целостные слова)
print('Пятый этап обработки...')
#Обработка изображения и нахождение контуров
gray = cv2.cvtColor(output_2, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (15, 15), 0)
gray = cv2.Canny(gray, 50, 50)
ret, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img_4 = img_3.copy()

for idx, contour in enumerate(contours):
    # Получение координат найденных контуров
    (x, y, w, h) = cv2.boundingRect(contour)
    if w * h > 2000:
        if w * h < 30000 and (w / h > 3 or h / w > 3):
            cv2.rectangle(img_4, (x, y), (x+w, y+h), (255, 255, 255), cv2.FILLED)

            img_cropped = img[y+10:y + h-10, x+10:x + w-10]

            # Создание массивов из набора каждого цвета в контуре
            b_array = np.array(img_cropped[:, :, 0]).flatten()
            g_array = np.array(img_cropped[:, :, 1]).flatten()
            r_array = np.array(img_cropped[:, :, 2]).flatten()

            # Определение компонент доминирующего цвета в контуре
            average_blue = needed_color(b_array)
            average_green = needed_color(g_array)
            average_red = needed_color(r_array)

            # Рисование фона для каждой буквы на чистом холсте
            cv2.rectangle(img_clear, (x - 5, y - 5), (x + w + 5, y + h + 5),
                          (255 - average_blue, 255 - average_green, 255 - average_red), -1)


result_4 = Image.fromarray(img_4)
result_4.save('result_4.png')

#### ЧАСТЬ 2 ПРОГРАММЫ (Обведение наиболее проблемных слов)

### ШЕСТОЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Обведение слов и исключение найденных)
print('Шестой этап обработки...')
img = cv2.imread('result_4.png')
img_res = img.copy()
image_copy = img.copy()


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
#Создание структурного элемента для более четкого распознавания слов
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
# Увеличение границ областей пикселей
dilated = cv2.dilate(binary, kernel_dilate, iterations=2)
#Создание элемента закрытия пропусков между буквами
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
#Выполнение закрытия пропусков между буквами
morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
#Нахождение контуров
contours_1, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
image_copy = img.copy()

for contour in contours_1:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)

    img_cropped = img[y+10:y + h-10, x+10:x + w-10]

    # Создание массивов из набора каждого цвета в контуре
    b_array = np.array(img_cropped[:, :, 0]).flatten()
    g_array = np.array(img_cropped[:, :, 1]).flatten()
    r_array = np.array(img_cropped[:, :, 2]).flatten()

    # Определение компонент доминирующего цвета в контуре
    average_blue = needed_color(b_array)
    average_green = needed_color(g_array)
    average_red = needed_color(r_array)

    # Рисование фона для каждой буквы на чистом холсте
    cv2.rectangle(img_clear, (x - 5, y - 5), (x + w + 5, y + h + 5),
                  (255 - average_red, 255 - average_green, 255 - average_blue), -1)


### СЕДЬМОЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Обведение слов и исключение найденных)
print('Седьмой этап обработки...')
gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
#Создание структурного элемента для более четкого распознавания слов
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
# Увеличение границ областей пикселей
dilated = cv2.dilate(binary, kernel_dilate, iterations=2)
#Создание элемента закрытия пропусков между буквами
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
#Выполнение закрытия пропусков между буквами
morph = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
#Нахождение контуров
contours_2, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours_2:
    (x, y, w, h) = cv2.boundingRect(contour)
    if w * h < 50000:
        cv2.rectangle(image_copy, (x, y), (x + w, y + h), (255, 255, 255), cv2.FILLED)

        img_cropped = img[y+10:y + h-10, x+10:x + w-10]

        # Создание массивов из набора каждого цвета в контуре
        b_array = np.array(img_cropped[:, :, 0]).flatten()
        g_array = np.array(img_cropped[:, :, 1]).flatten()
        r_array = np.array(img_cropped[:, :, 2]).flatten()

        # Определение компонент доминирующего цвета в контуре
        average_blue = needed_color(b_array)
        average_green = needed_color(g_array)
        average_red = needed_color(r_array)

        # Рисование фона для каждой буквы на чистом холсте
        cv2.rectangle(img_clear, (x - 5, y - 5), (x + w + 5, y + h + 5),
                      (255 - average_red, 255 - average_green, 255 - average_blue), -1)


### ВОСЬМОЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Добавление черных квадратов на изображение)
print('Восьмой этап обработки...')
gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
gray = cv2.Canny(gray, 80, 80)
contours_3, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours_3:
    x, y, w, h = cv2.boundingRect(contour)
    if w * h > 1250 and w * h < 50000:
        cv2.rectangle(image_copy, (x, y - 5), (x + w - 2, y + h + 5), (0, 0, 0), cv2.FILLED)




### ДЕВЯТЫЙ ЭТАП ОБВЕДЕНИЯ СЛОВ (Нахождение контуров вокруг черных квадратов и обведение)
print('Девятый этап обработки...')
gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (13, 13), 0)
gray = cv2.Canny(gray, 70, 70)
ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours_4, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

for contour in contours_4:
    (x, y, w, h) = cv2.boundingRect(contour)
    if w * h > 1250 and w * h < 50000:
        img_cropped = img[y+10:y + h-10, x+10:x + w-10]

        # Создание массивов из набора каждого цвета в контуре
        b_array = np.array(img_cropped[:, :, 0]).flatten()
        g_array = np.array(img_cropped[:, :, 1]).flatten()
        r_array = np.array(img_cropped[:, :, 2]).flatten()

        # Определение компонент доминирующего цвета в контуре
        average_blue = needed_color(b_array)
        average_green = needed_color(g_array)
        average_red = needed_color(r_array)


        # Рисование фона для каждой буквы на чистом холсте
        cv2.rectangle(img_clear, (x - 5, y - 5), (x + w + 5, y + h + 5),
                      (255 - average_red, 255 - average_green, 255 - average_blue), -1)



#Наложение двух картинок
print('Выполняется наложение картинок и их сохранение...')
background = img_clear
foreground = cv2.imread('words.png')

#Уравнивание размера двух картинок
foreground = cv2.resize(foreground, (background.shape[1], background.shape[0]))

#Создание ограничений по удалению белого цвета и светлых тонов
lower_white = np.array([201, 201, 201])
upper_white = np.array([255, 255, 255])

#Формирование маски
mask = cv2.inRange(foreground, lower_white, upper_white)
#Формирование инвертированной маски
mask_inv = cv2.bitwise_not(mask)
#Вырезание букв с изображения
fg_no_bg = cv2.bitwise_and(foreground, foreground, mask=mask_inv)
#Вырезание отверстий под буквы на холсте с квадратами
bg_with_hole = cv2.bitwise_and(background, background, mask=mask)
#Наложение картинок друг на друга
result = cv2.add(bg_with_hole, fg_no_bg)
cv2.imwrite('result.png', result)