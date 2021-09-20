import PIL
import PIL.Image as Image
import matplotlib.pylab as plt
import numpy as np
import cv2
import os

image = Image.open('WGolf.jpg').convert('L')
image = np.array(image) / 255.0  # Диапазон яркостей — [0, 1]
img = image[400:1200, 50:850]
jet = Image.open('jet.jpg').convert('L')
jet = np.array(jet) / 255.0  # Диапазон яркостей — [0, 1]


def createmd4(N, M):
    # создаем матрицу вейвлета d4 для строк изображения

    c1 = (1 + np.sqrt(3)) / (4 * np.sqrt(2))
    c2 = (3 + np.sqrt(3)) / (4 * np.sqrt(2))
    c3 = (3 - np.sqrt(3)) / (4 * np.sqrt(2))
    c4 = (1 - np.sqrt(3)) / (4 * np.sqrt(2))

    md4row = np.zeros((M, M))
    for i in range(0, M, 2):
        md4row[i, i] = c1
        md4row[i, i + 1] = c2
        md4row[i, ((i + 2) % M)] = c3
        md4row[i, ((i + 3) % M)] = c4
        md4row[i + 1, i] = c4
        md4row[i + 1, i + 1] = -c3
        md4row[i + 1, ((i + 2) % M)] = c2
        md4row[i + 1, ((i + 3) % M)] = -c1

    # создаем матрицу вейвлета d4 для столбцов изображения
    md4column = np.zeros((N, N))
    for i in range(0, N, 2):
        md4column[i, i] = c1
        md4column[i, i + 1] = c2
        md4column[i, ((i + 2) % N)] = c3
        md4column[i, ((i + 3) % N)] = c4
        md4column[i + 1, i] = c4
        md4column[i + 1, i + 1] = -c3
        md4column[i + 1, ((i + 2) % N)] = c2
        md4column[i + 1, ((i + 3) % N)] = -c1

    return md4row, md4column


def transfd4(img, depth, threshold=0):
    N, M = img.shape  # N - количество строк, M - количество столбцов

    # ограничение на глубину преобразования по размеру изображения
    if M < 4 or N < 4:
        return img

    # проверка размеров входного изображения на четность
    # при необходимости последние значения откидываются
    if N % 2 == 1:
        N -= 1
        img = np.delete(img, N, 0)
    if M % 2 == 1:
        M -= 1
        img = np.delete(img, M, 1)

    # создаем матрицы вейвлета d4 для строк и столбцов изображения
    md4row, md4column = createmd4(N, M)

    Hout = np.zeros(img.shape)

    # проход по строкам изображения
    for i in range(N):
        Hout[i] = np.dot(md4row, img[i])

    # для разделения значений используем zip
    # НЧ коэффициенты в левую часть строки, ВЧ в правую
    collector = range(0, M, 2)
    spreader = range(0, M // 2)
    Hout2 = np.zeros(img.shape)
    for m in range(N):
        for i, j in zip(collector, spreader):
            Hout2[m, j] = Hout[m, i]
            Hout2[m, j + M // 2] = Hout[m, i + 1]

    # проход по столбцам
    Hout3 = np.zeros(img.shape)
    for i in range(M):
        Hout3[:, i] = np.dot(md4column, Hout2[:, i])

    # для разделения значений используем zip
    # НЧ коэффициенты в левую часть строки, ВЧ в правую
    collector2 = range(0, N, 2)
    spreader2 = range(0, N // 2)
    Hout4 = np.zeros(img.shape)
    for m in range(M):
        for i, j in zip(collector2, spreader2):
            Hout4[j, m] = Hout3[i, m]
            Hout4[j + N // 2, m] = Hout3[i + 1, m]

    # отбрасывание значений из высокочастотных областей
    if threshold != 0:
        H = Hout4[0: N // 2, M // 2: M]
        th = threshold * np.amax(H)
        H = np.where(H >= th, H, 0)
        Hout4[0: N // 2, M // 2: M] = H
        V = Hout4[N // 2: N, 0: M // 2]
        th = threshold * np.amax(V)
        V = np.where(V >= th, V, 0)
        Hout4[N // 2: N, 0: M // 2] = V
        D = Hout4[N // 2: N, M // 2: M]
        th = threshold * np.amax(D)
        D = np.where(D >= th, D, 0)
        Hout4[N // 2: N, M // 2: M] = D

    # рекурсивный вызов функции
    if depth != 1:
        newDepth = depth - 1
        newImg = Hout4[0: N // 2, 0: M // 2]
        LL = transfd4(newImg, newDepth, threshold=threshold)
        Hout4[0: LL.shape[0], 0: LL.shape[1]] = LL

    return Hout4


def reversetransfd4(img, depth):
    N, M = img.shape  # N - количество строк, M - количество столбцов

    # проверка размеров входного изображения на четность
    # при необходимости последние значения откидываются
    if N % 2 == 1:
        N -= 1
        img = np.delete(img, N, 0)
    if M % 2 == 1:
        M -= 1
        img = np.delete(img, M, 1)

    # рекурсивный вызов функции
    if depth != 1:
        newDepth = depth - 1
        newImg = img[0: N // 2, 0: M // 2]
        LL = reversetransfd4(newImg, newDepth)
        img[0: LL.shape[0], 0: LL.shape[1]] = LL

    # создаем обратные матрицы вейвлета d4 для строк и столбцов изображения
    md4row, md4column = createmd4(N, M)
    imd4row = np.transpose(md4row)
    imd4column = np.transpose(md4column)

    # сортировка значений, approximation на i место (0, 2, 4...),
    # difference на i+1 место (1, 3, 5...)
    # проход по стобцам
    rHout = np.zeros(img.shape)
    for m in range(M):
        collector = range(0, N // 2)
        spreader = range(0, N, 2)
        for i, j in zip(collector, spreader):
            rHout[j, m] = img[i, m]
            rHout[j + 1, m] = img[i + (N // 2), m]

    # умножение изображения на обратную матрицу вейвлета d4
    rHout2 = np.zeros(img.shape)
    for m in range(M):
        rHout2[:, m] = np.dot(imd4column, rHout[:, m])

    # сортировка значений в строках
    rHout3 = np.zeros(img.shape)
    for m in range(N):
        collector = range(0, M // 2)
        spreader = range(0, M, 2)
        for i, j in zip(collector, spreader):
            rHout3[m, j] = rHout2[m, i]
            rHout3[m, j + 1] = rHout2[m, i + (M // 2)]

    # умножение изображения на обратную матрицу вейвлета d4
    rHout4 = np.zeros(img.shape)
    for m in range(N):
        rHout4[m] = np.dot(imd4row, rHout3[m])

    return rHout4


counter = 1
os.makedirs('.\\d4_exp2')
for depth in range(1, 4):
    for threshold in range(1, 10, 2):
        threshold *= 0.1

        forwD4 = transfd4(jet, depth, threshold=threshold)
        forwD4_ = forwD4 * 255
        resNules = (forwD4.shape[0] * forwD4.shape[1]) - np.count_nonzero(forwD4)
        percentNules = resNules / (forwD4.shape[0] * forwD4.shape[1]) * 100
        name = '.\\d4_exp2\\' + str(counter) + ' jet forwD4 depth=' + \
               str(depth) + ' threshold=' + str("%.2f" % threshold) + \
               ' resNules=' + str(resNules) + ' percentNules=' + \
               str("%.2f" % percentNules) + '.jpeg'
        cv2.imwrite(name, forwD4_)
        print(counter)
        counter += 1

        backD4 = reversetransfd4(forwD4, depth)
        backD4_ = backD4 * 255
        name2 = '.\\d4_exp2\\' + str(counter) + ' jet backD4 depth=' + \
                str(depth) + ' threshold=' + str("%.2f" % threshold) + \
                ' resNules=' + str(resNules) + ' percentNules=' + \
                str("%.2f" % percentNules) + '.jpeg'
        cv2.imwrite(name2, backD4_)
        print(counter)
        counter += 1
