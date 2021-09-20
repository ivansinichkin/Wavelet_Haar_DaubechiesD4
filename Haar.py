import PIL
import PIL.Image as Image
import matplotlib.pylab as plt
import numpy as np
import cv2
import os

image = Image.open('WGolf.jpg').convert('L')
image = np.array(image) / 255.0  # Диапазон яркостей — [0, 1]
jet = Image.open('jet.jpg').convert('L')
jet = np.array(jet) / 255.0  # Диапазон яркостей — [0, 1]
img = image[400:1200, 50:850]


def transfHaar(img, depth, threshold=0):
    N, M = img.shape  # N - количество строк, M - количество столбцов

    # ограничение на глубину преобразования по размеру изображения
    if M <= 3 or N <= 3:
        return img

    # проверка размеров входного изображения на четность
    # при необходимости последние значения откидываются
    if N % 2 == 1:
        N -= 1
        img = np.delete(img, N, 0)
    if M % 2 == 1:
        M -= 1
        img = np.delete(img, M, 1)

    # Создаем матрицу для преобразования Хаара
    mHrow = np.zeros((M, M))
    for i in range(0, M, 2):
        mHrow[i, i] = (1 / np.sqrt(2))
        mHrow[i + 1, i + 1] = (1 / np.sqrt(2))
        mHrow[i, i + 1] = (1 / np.sqrt(2))
        mHrow[i + 1, i] = -(1 / np.sqrt(2))

    mHcolumn = np.zeros((N, N))
    for i in range(0, N, 2):
        mHcolumn[i, i] = (1 / np.sqrt(2))
        mHcolumn[i + 1, i + 1] = (1 / np.sqrt(2))
        mHcolumn[i, i + 1] = (1 / np.sqrt(2))
        mHcolumn[i + 1, i] = -(1 / np.sqrt(2))

    Hout = np.zeros(img.shape)
    # проход по строкам изображения
    for i in range(N):
        Hout[i] = np.dot(mHrow, img[i])

    # для разделения значений используем zip
    approximation = range(0, M, 2)
    difference = range(0, M // 2)
    Hout_ = np.zeros(img.shape)
    for m in range(N):
        for i, j in zip(approximation, difference):
            Hout_[m, j] = Hout[m, i]
            Hout_[m, j + M // 2] = Hout[m, i + 1]

    # проход по столбцам
    Hout2 = np.zeros(img.shape)
    for i in range(M):
        Hout2[:, i] = np.dot(mHcolumn, Hout_[:, i])
    approximation2 = range(0, N, 2)
    difference2 = range(0, N // 2)
    Hout2_ = np.zeros(img.shape)
    for m in range(M):
        for i, j in zip(approximation2, difference2):
            Hout2_[j, m] = Hout2[i, m]
            Hout2_[j + N // 2, m] = Hout2[i + 1, m]

    # отбрасывание значений из высокочастотных областей
    if threshold != 0:
        H = Hout2_[0: N // 2, M // 2: M]
        th = threshold * np.amax(H)
        H = np.where(H >= th, H, 0)
        Hout2_[0: N // 2, M // 2: M] = H
        V = Hout2_[N // 2: N, 0: M // 2]
        th = threshold * np.amax(V)
        V = np.where(V >= th, V, 0)
        Hout2_[N // 2: N, 0: M // 2] = V
        D = Hout2_[N // 2: N, M // 2: M]
        th = threshold * np.amax(D)
        D = np.where(D >= th, D, 0)
        Hout2_[N // 2: N, M // 2: M] = D

    # рекурсивный вызов функции
    if depth != 1:
        newDepth = depth - 1
        newImg = Hout2_[0: N // 2, 0: M // 2]
        LL = transfHaar(newImg, newDepth, threshold=threshold)
        Hout2_[0: LL.shape[0], 0: LL.shape[1]] = LL

    return Hout2_


def reverseTransfHaar(img, depth):
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
        LL = reverseTransfHaar(newImg, newDepth)
        img[0: LL.shape[0], 0: LL.shape[1]] = LL

    # Создаем матрицу для обратного преобразования Хаара
    imHrow = np.zeros((M, M))
    for i in range(0, M, 2):
        imHrow[i, i] = (1 / np.sqrt(2))
        imHrow[i + 1, i + 1] = (1 / np.sqrt(2))
        imHrow[i, i + 1] = -(1 / np.sqrt(2))
        imHrow[i + 1, i] = (1 / np.sqrt(2))

    imHcolumn = np.zeros((N, N))
    for i in range(0, N, 2):
        imHcolumn[i, i] = (1 / np.sqrt(2))
        imHcolumn[i + 1, i + 1] = (1 / np.sqrt(2))
        imHcolumn[i, i + 1] = -(1 / np.sqrt(2))
        imHcolumn[i + 1, i] = (1 / np.sqrt(2))

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

    # умножение изображения на обратную матрицу Хаара
    rHout2 = np.zeros(img.shape)
    for m in range(M):
        rHout2[:, m] = np.dot(imHcolumn, rHout[:, m])

    # сортировка значений в строках
    rHout3 = np.zeros(img.shape)
    for m in range(N):
        collector = range(0, M // 2)
        spreader = range(0, M, 2)
        for i, j in zip(collector, spreader):
            rHout3[m, j] = rHout2[m, i]
            rHout3[m, j + 1] = rHout2[m, i + (M // 2)]

    # умножение изображения на обратную матрицу Хаара
    rHout4 = np.zeros(img.shape)
    for m in range(N):
        rHout4[m] = np.dot(imHrow, rHout3[m])

    return rHout4


counter = 1
# os.makedirs('.\\exp1')
for depth in range(1, 4):
    for threshold in range(1, 10, 2):
        threshold *= 0.1

        forwH = transfHaar(jet, depth, threshold=threshold)
        forwH_ = forwH * 255
        resNules = (forwH.shape[0] * forwH.shape[1]) - np.count_nonzero(forwH)
        percentNules = resNules / (forwH.shape[0] * forwH.shape[1]) * 100
        name = '.\\haar_exp2\\' + str(counter) + ' jet forwH depth=' + \
               str(depth) + ' threshold=' + str("%.2f" % threshold) + \
               ' resNules=' + str(resNules) + ' percentNules=' + \
               str("%.2f" % percentNules) + '.jpeg'
        cv2.imwrite(name, forwH_)
        print(counter)
        counter += 1

        backH = reverseTransfHaar(forwH, depth)
        backH_ = backH * 255
        name2 = '.\\haar_exp2\\' + str(counter) + ' jet backH depth=' + \
                str(depth) + ' threshold=' + str("%.2f" % threshold) + \
                ' resNules=' + str(resNules) + ' percentNules=' + \
                str("%.2f" % percentNules) + '.jpeg'
        cv2.imwrite(name2, backH_)
        print(counter)
        counter += 1
