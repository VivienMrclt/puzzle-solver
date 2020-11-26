import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.signal import convolve2d as convolve2d
from scipy.ndimage import gaussian_filter
import matplotlib.animation as animation
from numpy import random

truth = mpimg.imread('truth.jpg')
pieces = mpimg.imread('pieces0.jpg')
f = 6

fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# pieces = rgb2gray(pieces)
axes[0, 0].imshow(pieces, cmap=plt.get_cmap("gray"))

def convolution2d(image, kernel):
    img = np.zeros(image.shape)
    if len(image.shape) == 3:
        for i in range(3):
            img[:, :, i] = convolve2d(image[:, :, i], kernel, "same", "symm")
    else:
        img[:, :] = convolve2d(image[:, :], kernel, "same", "symm")
    return img

def gauss(image):
    img = np.zeros(image.shape)
    if len(image.shape) == 3:
        for i in range(3):
            img[:, :, i] = gaussian_filter(image[:, :, i], sigma=1.96)
    else:
        img[:, :] = gaussian_filter(image[:, :], sigma=1.96)
    return img


Hp = pieces.shape[0]
Lp = pieces.shape[1]
hp = Hp // f
lp = Lp // f


smoothedPieces = gauss(np.array(pieces, dtype=np.uint64)/255)
# plt.imshow(smoothedPieces/255)
# plt.show()

#convolution2d(pieces, np.array([[2, 4, 5, 4, 2], [4, 9, 12, 9, 4], [5, 12, 15, 12, 5], [4, 9, 12, 9, 4], [2, 4, 5, 4, 2]])/159)

s5 = np.array([[-2./8, -1./5, 0, 1./5, 2./8],
               [-2./5, -1./2, 0, 1./2, 2./5],
               [-2./4, -1./1, 0, 1./1, 2./4],
               [-2./5, -1./2, 0, 1./2, 2./5],
               [-2./8, -1./5, 0, 1./5, 2./8]])

s7 = np.array([[-130, -120,  -78, 0,  78, 120, 130],
               [-180, -195, -156, 0, 156, 195, 180],
               [-234, -312, -390, 0, 390, 312, 234],
               [-260, -390, -780, 0, 780, 390, 260],
               [-234, -312, -390, 0, 390, 312, 234],
               [-180, -195, -156, 0, 156, 195, 180],
               [-130, -120,  -78, 0,  78, 120, 130]], dtype=np.float) / 780

s9 = np.array([[-16575, -15912, -13260,  -7800, 0,   7800, 13260, 15912, 16575],
               [-21216, -22100, -20400, -13260, 0,  13260, 20400, 22100, 21216],
               [-26520, -30600, -33150, -26520, 0,  26520, 33150, 30600, 26520],
               [-31200, -39780, -53040, -66300, 0,  66300, 53040, 39780, 31200],
               [-33150, -44200, -66300,-132600, 0, 132600, 66300, 44200, 33150],
               [-31200, -39780, -53040, -66300, 0,  66300, 53040, 39780, 31200],
               [-26520, -30600, -33150, -26520, 0,  26520, 33150, 30600, 26520],
               [-21216, -22100, -20400, -13260, 0,  13260, 20400, 22100, 21216],
               [-16575, -15912, -13260,  -7800, 0,   7800, 13260, 15912, 16575]], dtype=np.float) / 132600


Gx = convolution2d(smoothedPieces, s9)
Gy = convolution2d(smoothedPieces, s9.T)

# Gx = convolution2d(smoothedPieces, np.array([[-3, -1, 0, 1, 3]]))
# Gy = convolution2d(smoothedPieces, np.array([[-3], [-1], [0], [1], [3]]))
S = Gx + Gy * 1j

mod = np.abs(S)
mod = np.average(mod, axis=2)

# modLog = np.log(1 + mod) / np.log(1 + np.quantile(mod, 0.90))

# modSmoothed = gaussian_filter(modLog, sigma=5)

d5 = np.array([[1, 2, 5, 2, 1],
               [1, 2, 5, 2, 1],
               [1, 2, 5, 2, 1],
               [1, 2, 5, 2, 1],
               [1, 2, 5, 2, 1]], dtype=np.float) / 55

d9 = np.array([[-1, 1, 3, 9, 15, 9, 3, 1, -1],
               [-1, 1, 3, 9, 15, 9, 3, 1, -1],
               [-1, 1, 3, 9, 15, 9, 3, 1, -1],
               [-1, 1, 3, 9, 15, 9, 3, 1, -1],
               [-1, 1, 3, 9, 15, 9, 3, 1, -1],
               [-1, 1, 3, 9, 15, 9, 3, 1, -1],
               [-1, 1, 3, 9, 15, 9, 3, 1, -1],
               [-1, 1, 3, 9, 15, 9, 3, 1, -1],
               [-1, 1, 3, 9, 15, 9, 3, 1, -1]], dtype=np.float) / (39*9)

# modSmoothed = gaussian_filter(mod, sigma=2)

axes[1, 0].imshow(mod , cmap='gray')

# Dx = convolution2d(mod, d9.T)
# Dy = convolution2d(mod, d9)
#
# S = Dx + Dy * 1j
#
# mod = np.abs(S)

#modLog = 1 / ( 1 + np.exp((np.quantile(mod, 0.95) - mod)))


print("Filter applied")
resized = np.median(mod[:hp*f, :lp*f].reshape(hp, f, lp, f), axis=(1, 3))


axes[1, 1].imshow(resized.repeat(f ** 2).reshape(hp, lp, f, f).swapaxes(1, 2).reshape(hp * f, lp * f) , cmap='gray')


def marcher(img, limit):
    mh, ml = img.shape
    border = np.zeros(img.shape, dtype=np.bool)
    border[0, :] = True
    border[mh - 1, :] = True
    border[:, 0] = True
    border[:, ml - 1] = True
    mask = np.copy(border);
    maskDelay = np.zeros(border.shape);
    maskTemp = np.logical_not(border);
    borderTemp1 = np.zeros(border.shape, dtype=np.bool);
    borderTemp2 = np.zeros(border.shape, dtype=np.bool);
    borderTemp3 = np.zeros(border.shape, dtype=np.bool);

    step = 0
    while(np.sum(np.logical_xor(mask, maskTemp)) > 0):
        global axes
        maskTemp = np.copy(mask)
        stillBorder = np.copy(border)
        for dh, dl in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
            bin = random.rand(mh, ml)*np.quantile(img, limit) > img
            candidates = np.zeros(border.shape, dtype=np.bool)
            candidates[max(0, dh):min(mh, mh+dh), max(0, dl):min(ml, ml+dl)] = border[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)]
            candidates = np.logical_and(candidates, np.logical_not(mask))
            newBorder = np.logical_and(candidates, bin)
            stillBorder[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)] = np.logical_or(stillBorder[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)], newBorder[max(0, dh):min(mh, mh+dh), max(0, dl):min(ml, ml+dl)])
            stillBorder = np.logical_or(border, newBorder)
            border = np.logical_or(border, newBorder)
            mask = np.logical_or(mask, newBorder)
            maskDelay[newBorder] = step
            step+= 1
        border = np.logical_and(border, stillBorder)
        border = np.logical_and(border, np.logical_not(borderTemp3))
        borderTemp3 = np.copy(borderTemp2)
        borderTemp2 = np.copy(borderTemp1)
        borderTemp1 = np.copy(border)
    maskDelay[np.logical_not(mask)] = np.max(maskDelay)
    axes[1, 2].imshow((random.rand(mh, ml)*np.quantile(img, limit) > img).repeat(f ** 2).reshape(hp, lp, f, f).swapaxes(1, 2).reshape(hp * f, lp * f))

    return mask, maskDelay

# x = np.linspace(0, np.quantile(mod, 0.90), 100)
# # x = np.linspace(0.2, 0.6, 100)
# y = [np.sum(marcher(resized, limit)) for limit in x]
#
# plt.plot(x, y)
# plt.show()

mask, maskDelay = marcher(resized, 0.97)

axes[0, 1].imshow(mask.repeat(f ** 2).reshape(hp, lp, f, f).swapaxes(1, 2).reshape(hp * f, lp * f) , cmap='gray')
axes[0, 2].imshow(maskDelay.repeat(f ** 2).reshape(hp, lp, f, f).swapaxes(1, 2).reshape(hp * f, lp * f))
plt.show()

#
# fig = plt.figure()
# im = plt.imshow(mask, cmap='gray', animated=True)

#
# def updatefig(*args):
#     global step, border, mask, modLog, im, borderTemp, bin
#     print("step ", step, ", len(border) = ", np.sum(border))
#     stillBorder = np.copy(border)
#     for dh, dl in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
#         candidates = np.zeros(border.shape, dtype=np.bool)
#         candidates[max(0, dh):min(mh, mh+dh), max(0, dl):min(ml, ml+dl)] = border[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)]
#         candidates = np.logical_and(candidates, np.logical_not(mask))
#         newBorder = np.logical_and(candidates, bin)
#         stillBorder[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)] = np.logical_or(stillBorder[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)], newBorder[max(0, dh):min(mh, mh+dh), max(0, dl):min(ml, ml+dl)])
#         stillBorder = np.logical_or(border, newBorder)
#         border = np.logical_or(border, newBorder)
#         mask = np.logical_or(mask, newBorder)
#     border = np.logical_and(border, stillBorder)
#     border = np.logical_and(border, np.logical_not(borderTemp))
#     borderTemp = np.copy(border)
#     im.set_array(mask)
#     step += 1
#     return im,
#
# ani = animation.FuncAnimation(fig, updatefig, interval=2, blit=True)
# plt.show()

# while (np.logical_xor(border, borderTemp).any()):
# while(np.sum(border) > 0):
#     print("step ", step, ", len(border) = ", np.sum(border))
#     stillBorder = np.copy(border)
#     bin = modLog < random.rand(mh, ml)
#     for dh, dl in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]:
#         candidates = np.zeros(border.shape, dtype=np.bool)
#         candidates[max(0, dh):min(mh, mh+dh), max(0, dl):min(ml, ml+dl)] = border[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)]
#         candidates = np.logical_and(candidates, np.logical_not(mask))
#         newBorder = np.logical_and(candidates, bin)
#         stillBorder[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)] = np.logical_or(stillBorder[max(0, -dh):min(mh, mh-dh), max(0, -dl):min(ml, ml-dl)], newBorder[max(0, dh):min(mh, mh+dh), max(0, dl):min(ml, ml+dl)])
#         stillBorder = np.logical_or(border, newBorder)
#         border = np.logical_or(border, newBorder)
#         mask = np.logical_or(mask, newBorder)
#     border = np.logical_and(border, stillBorder)
#     border = np.logical_and(border, np.logical_not(borderTemp))
#     borderTemp = np.copy(border)
#     step += 1
#
# im = plt.imshow(mask, cmap='gray')
# plt.show()


(H, L, _) = truth.shape

l = 6
h = 5

meanColors = np.zeros((h, l, 3))

rh = H // h;
rl = L // l;

for i in range(h):
    for j in range(l):
        if i == h - 1:
            bh = H
        else:
            bh = (i+1) * rh + 1
        if j == l - 1:
            bl = L
        else:
            bl = (j+1) * rl + 1
        meanColors[i, j] = np.average(truth[i * rh: bh, j * rl: bl], axis=(0, 1))

# plt.imshow(meanColors/255)
# plt.show()
