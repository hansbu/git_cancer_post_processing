import os
import sys
import cv2
import csv
import numpy as np
import openslide
import math
import argparse

import matplotlib as mpl
mpl.use('TkAgg')
mpl.use('pdf')
import matplotlib.pylab as plt

from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap, BoundaryNorm
import multiprocessing





class XYLabelFile(object):
    """
    This class reads cvs file seperated by ' ', every line of which is like x y ...
    And convert it to x,y 2D array.
    The whole process is: 1)init 2)set width and height 3)extract
    """
    def __init__(self, file_path, skip_header=False):
        """ Read the file. """
        self.filePath = file_path
        self.skip_header = skip_header
        self.data = self.read_file(self.filePath)
        self.patchSize = 0
        self.width = 0
        self.height = 0
        self.extracted = None


    def setWidthHeightByOSlide(self, slide):
        self.width = slide.dimensions[0]
        self.height = slide.dimensions[1]

    def setWidthHeight(self, width, height):
        self.width = width
        self.height = height


    def read_file(self, file_path):
        """ Read prediction file into a numpy 2D array """
        return np.genfromtxt(file_path, delimiter=' ', skip_header=1 if self.skip_header else 0)

    def extract(self, indexes):
        """
        This function extracts the column of csv file you want.
        Args:
            indexes(list): The indexes of columns you want to extracted
        Retures:
            list: self.extracted. List of extracted 2D arrays.
        """
        uniqueX = np.unique(self.data[:, 0])
        self.patchSize = uniqueX[1] - uniqueX[0]
#         print("max x,y:", np.max(self.data[:, [0,1]], axis=0))
#         print(self.patchSize)
        rowIndex = np.logical_and(self.data[:,0] + self.patchSize / 2 < self.width,
                                  self.data[:,1] + self.patchSize / 2 < self.height)
        filteredData = self.data[rowIndex, :]

        xys_pred = ((filteredData[:, [0, 1]] + self.patchSize / 2) / self.patchSize - 1).astype(np.int)
#         print("oslide width & height:", self.width, self.height)
#         print("filteredData.shape", filteredData.shape)
#         print("data.shape:", self.data.shape)
#         print("index min x,y:", np.min(xys_pred, axis=0))
#         print("index max x,y:", np.max(xys_pred, axis=0))
        self.extracted = []
        shape = int(self.width // self.patchSize), int(self.height // self.patchSize)
#         print("small shape:", shape)
        for i in indexes:
            self.extracted.append(np.zeros(shape))

        for i in range(xys_pred.shape[0]):
            l = filteredData[i,:]
            x, y = xys_pred[i]
            for j, arr in zip(indexes, self.extracted):
                arr[x,y] = l[j]
        return self.extracted


    def extract_old(self, indexes):
        """
        This function extracts the column of csv file you want.
        This version do not filter
        Args:
            indexes(list): The indexes of columns you want to extracted
        Retures:
            list: self.extracted. List of extracted 2D arrays.
        """
        xys = self.data[:, [0,1]] # x, y
        mins = np.min(xys, axis=0)
        maxs = np.max(xys, axis=0)

        width = mins[0] + maxs[0]
        height = mins[1] + maxs[1]
        print("width:",width, "\t height:",height)

        uniqueX = np.unique(xys[:, 0])
        self.patchSize = width / uniqueX.size

        xys_pred = ((xys + self.patchSize / 2) / self.patchSize - 1).astype(np.int)

        shape = np.max(xys_pred,axis=0)+1

        self.extracted = []

        for i in indexes:
            self.extracted.append(np.zeros(shape))

        for i in range(xys_pred.shape[0]):
            l = self.data[i,:]
            x, y = xys_pred[i]
            for j, arr in zip(indexes, self.extracted):
                arr[x,y] = l[j]
        return self.extracted


class PredictionFile(XYLabelFile):
    def __init__(self, file_path, skip_header=False):
        super().__init__(file_path, skip_header)
        self.pred = None
        self.necr = None

    def get_pred_and_necr(self):
        if self.data.shape[1] > 3:
            self.pred, self.necr = self.extract([2, 3])
        else:
            self.pred = self.extract([2])
            self.necr = np.ones_like(self.pred, dtype=self.pred.dtype)
        return self.pred, self.necr, self.patchSize

    def get_labeled_im(self):
        return self.get_pred_and_necr()


class ColorFile(XYLabelFile):
    def __init__(self, file_path):
        super().__init__(file_path)
        self.whiteness = None
        self.blackness = None
        self.redness = None

    def get_color_channels(self):
        self.whiteness, self.blackness, self.redness = self.extract([2, 3, 4])
        return self.whiteness, self.blackness, self.redness

    def get_whiteness_im(self):
        return self.get_color_channels()


class HeatMap(object):

    def __init__(self, rootFolder, skip_first_line_pred):
        self.rootFolder = rootFolder
        self.width = 0
        self.height = 0
        self.skip_first_line_pred = skip_first_line_pred

#     def __init__(self, rootFolder, slideId):
#         self.__init__(rootFolder)
#         self.slideIddeId = slideId

#         predictionFileName = 'prediction-'+slideId
#         colorFileName = 'color-'+slideId
#         self.heatmap = self.getHeatMap(os.path.join(self.rootFolder, predictionFileName),
#                                os.path.join(self.rootFolder, colorFileName))

    def setWidthHeightByOSlide(self, slide):
        self.width = slide.dimensions[0]
        self.height = slide.dimensions[1]

    def getHeatMapByID(self, slideId):
        """
        Args:
            slideId(str): id of svs file, like 'TCGA-3C-AALI-01Z-00-DX1'.
        """
        predictionFileName = 'prediction-'+slideId
        colorFileName = 'color-'+slideId
        self.heatmap = self.getHeatMap(os.path.join(self.rootFolder, predictionFileName),
                               os.path.join(self.rootFolder, colorFileName))
        return self.heatmap

    def getHeatMap(self, predPath, colorPath):
        """
        Args:
            predPath(str): must be full path.
            colorPath(str): must be full path.
        """
        predictionFile = PredictionFile(predPath, self.skip_first_line_pred)
        predictionFile.setWidthHeight(self.width, self.height)
        pred, necr, patch_size = predictionFile.get_pred_and_necr()
#         print("pred.shape:",pred.shape)

        colorFile = ColorFile(colorPath)
        colorFile.setWidthHeight(self.width, self.height)
        whiteness, blackness, redness = colorFile.get_whiteness_im()

        image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        image[:, :, 0] = np.multiply(255* pred, (blackness>30).astype(np.float64), (redness<0.15).astype(np.float64))
        image[:, :, 1] = necr
        image[:, :, 2] = 255 * (cv2.GaussianBlur(whiteness, (5, 5), 0) > 12)
        out = image[:,:, [2, 1, 0]]
        out = np.transpose(out, (1, 0, 2))
        return out


class MergedHeatMap(object):
    """
    Merge cacer heatmap and lymp heatmap together.
    Input image arrays should be opencv type(BGR).
    """
    def __init__(self, cancerImg, lympImg):
        self.cancerImg = cancerImg
        self.lympImg = lympImg
        self.mergedHeatMap = self.merge(cancerImg, lympImg)

    def thresholding(self, array):
        threshold = 255 * 0.5
        out = np.zeros_like(array, dtype=np.uint8)
        out[array > threshold] = 255
        return out

    def merge(self, cancerImg, lympImg):
        cancerArray = self.thresholding(cancerImg[:, :, 2])

        up = int(math.ceil(lympImg.shape[0]/cancerArray.shape[0]))

        if up > 1:
            iml_u = np.zeros((cancerArray.shape[0] * up, cancerArray.shape[1] * up), dtype=np.float32)
            for x in range(cancerArray.shape[1]):
                for y in range(cancerArray.shape[0]):
                    iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = cancerArray[y, x]
            #cancerArray = iml_u.copy()
            cancerArray = iml_u.astype(np.uint8)

        smooth5 = cancerArray
        if np.max(smooth5) < 2:
            smooth5 = (smooth5*255).astype(np.uint8)

        smooth5 = cv2.resize(smooth5, (lympImg.shape[1], lympImg.shape[0]), interpolation=cv2.INTER_LINEAR)
        smooth5 = cv2.GaussianBlur(smooth5, (5, 5), 0)

        out = np.zeros_like(lympImg, dtype=np.uint8)

        for i in range(lympImg.shape[0]):
            for j in range(lympImg.shape[1]):
                b, g, r = lympImg[i, j]
                out[i, j] = np.array([192,192,192])
                is_tumor = smooth5[i, j] > 100
                is_lym = r > 100
                is_tisue = (b >= 0.5 * 255)
                # Tissue, Tumor, Lym
                if (not is_tumor) and (not is_lym): # BGR
                    if not is_tisue:
                        out[i, j] = np.array([255,255,255]) #White
                    else:
                        out[i, j] = np.array([192,192,192]) # Grey
                elif is_tumor and (not is_lym):
                    out[i, j] = np.array([0,255,255]) #Yellow
                elif (not is_tumor) and is_lym:
                    out[i, j] = np.array([0,0,200]) #Redish
                else:
                    out[i, j] = np.array([0,0,255]) #Red
        return out


class StagePredictionFile(XYLabelFile):
    def __init__(self, file_path, skip_header=False):
        super().__init__(file_path, skip_header)
        self.g3 = None
        self.g45 = None
        self.benign = None
        self.stroma = None
        self.pred = None

    def get_stage_prediction(self):
        self.g3, self.g45, self.benign = self.extract([2, 3, 4])
        self.pred = self.g3 + self.g45
        return self.g3, self.g45, self.benign

    def get_labeled_im(self):
        return self.get_stage_prediction()


class StagedTumorHeatMap(HeatMap):

    def __init__(self, rootFolder, skip_first_line_pred):
        super().__init__(rootFolder, skip_first_line_pred)


    def getHeatMapByID(self, slideId):
        """
        Args:
            slideId(str): id of svs file, like 'TCGA-3C-AALI-01Z-00-DX1'.
        """
        predictionFileName = prefix + slideId
        colorFileName = 'color-'+slideId
        self.heatmap = self.getHeatMap(os.path.join(self.rootFolder, predictionFileName),
                               os.path.join(self.rootFolder, colorFileName))
        return self.heatmap

    def getHeatMap(self, stagePredPath, colorPath):
        predictionFile = StagePredictionFile(stagePredPath, self.skip_first_line_pred)
        predictionFile.setWidthHeight(self.width, self.height)
        predictionFile.get_stage_prediction()

        pred = predictionFile.pred
#         print("StagedTumorPred.shape:",pred.shape)

        colorFile = ColorFile(colorPath)
        colorFile.setWidthHeight(self.width, self.height)
        whiteness, blackness, redness = colorFile.get_whiteness_im()

        image = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)

        image[:, :, 0] = np.multiply(255* pred, (blackness>30).astype(np.float64), (redness<0.15).astype(np.float64))
        #image[:, :, 1] = necr
        image[:, :, 2] = 255 * (cv2.GaussianBlur(whiteness, (5, 5), 0) > 12)
        out = image[:,:, [2, 1, 0]]
        out = np.transpose(out, (1, 0, 2))

        self.predictionFile = predictionFile
        self.colorFile = colorFile
        self.tissue = image[:, :, 2]
        return out

    def getTumorClassificationMap(self):
        predictionFile = self.predictionFile
        stackedArray = np.stack([predictionFile.pred,
                                 predictionFile.benign],  # stroma
                                axis=2)
        #         print("stackedArray.shape",stackedArray.shape)
        classification = np.argmax(stackedArray, axis=2)
        mask = np.sum(stackedArray, axis=2) > 0.1

        colorArray = np.array([  # rgb array
            [255, 255,   0],  # tumor yellow
            [  0,   0, 255],  # benign blue
            [192, 192, 192]  # stroma gray # not used
        ])
        self.tumorColorArray = colorArray

        image = np.ones((stackedArray.shape[0], stackedArray.shape[1], 3), dtype=np.uint8)
        image = image * 192  # all gray
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if self.tissue[i, j] > 100 and mask[i, j]:
                    image[i, j] = colorArray[classification[i, j], :]

        image = np.transpose(image, (1, 0, 2))
        image = image[:, :, [2, 1, 0]]  # convert bgr
        self.tumorClassificationMap = image
        return image

    def getStageClassificationMap(self):
        predictionFile = self.predictionFile
        stackedArray = np.stack([predictionFile.g3, predictionFile.g45,
                               predictionFile.benign], # stroma
                               axis=2)
#         print("stackedArray.shape",stackedArray.shape)
        classification = np.argmax(stackedArray, axis=2)
        mask = np.sum(stackedArray, axis=2) > 0.1

        colorArray = np.array([ # rgb array
            [  0, 255,   0], # g3 green
            [255, 165,   0], # g45 orange
            [  0,   0, 255], # benign blue
            [192, 192, 192]  # stroma gray # not used
        ])
        self.stageColorArray = colorArray


        image = np.ones((stackedArray.shape[0], stackedArray.shape[1], 3), dtype=np.uint8)
        image = image * 192  # all gray
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if self.tissue[i, j] > 100 and mask[i, j]:
                    image[i, j] = colorArray[classification[i, j], :]

        image = np.transpose(image, (1, 0, 2))
        image = image[:,:, [2, 1, 0]] # convert bgr
        self.stageClassificationMap = image
        return image

    def thresholding(self, array):
        threshold = 255 * 0.5
        out = np.zeros_like(array, dtype=np.uint8)
        out[array > threshold] = 255
        return out

    def getStageClassificationTilMap(self, tilMap):
        cancerProb = self.predictionFile.pred
        cancerProb = np.transpose(cancerProb, (1, 0))
        lympImg = tilMap
        cancerArray = self.thresholding(cancerProb*255)

        up = int(math.ceil(lympImg.shape[0] / cancerArray.shape[0]))

        if up > 1:
            iml_u = np.zeros((cancerArray.shape[0] * up, cancerArray.shape[1] * up), dtype=np.float32)
            for x in range(cancerArray.shape[1]):
                for y in range(cancerArray.shape[0]):
                    iml_u[y * up:(y + 1) * up, x * up:(x + 1) * up] = cancerArray[y, x]
            # cancerArray = iml_u.copy()
            cancerArray = iml_u.astype(np.uint8)

        smooth5 = cancerArray
        if np.max(smooth5) < 2:
            smooth5 = (smooth5 * 255).astype(np.uint8)

        smooth5 = cv2.resize(smooth5, (lympImg.shape[1], lympImg.shape[0]), interpolation=cv2.INTER_LINEAR)
        smooth5 = cv2.GaussianBlur(smooth5, (5, 5), 0)

        print(np.max(smooth5))

        out = np.zeros_like(lympImg, dtype=np.uint8)

        for i in range(lympImg.shape[0]):
            for j in range(lympImg.shape[1]):
                b, g, r = lympImg[i, j]
                out[i, j] = np.array([192, 192, 192])
                is_tumor = smooth5[i, j] > 100
                is_lym = r > 100
                is_tisue = (b >= 0.5 * 255)
                # Tissue, Tumor, Lym
                if (not is_tumor) and (not is_lym):  # BGR
                    if not is_tisue:
                        out[i, j] = np.array([255, 255, 255])  # White
                    else:
                        out[i, j] = np.array([255, 0, 0])  # blue # original 192 gray
                elif is_tumor and (not is_lym):
                    # print("tumor & not lym")
                    out[i, j] = np.array([0, 255, 255])  # Yellow
                elif (not is_tumor) and is_lym:
                    out[i, j] = np.array([0, 0, 200])  # Redish
                else:
                    out[i, j] = np.array([0, 0, 255])  # Red
        return out


class FourPanelGleasonImage(object):
    def __init__(self, oslide, cancerImg, classificationMap, stageClassificationMap, savePath):
        self.oslide = oslide
        self.cancerImg = cancerImg
        self.classificationMap = classificationMap
        self.stageClassificationMap = stageClassificationMap
        self.savePath = savePath

    def saveImg(self):
        shape = (self.cancerImg.shape[1] * 2, self.cancerImg.shape[0] * 2)
        thumbnail = self.oslide.get_thumbnail(shape)

        cancerImg = self.cancerImg

        classificationMap = self.classificationMap[:, :, [2, 1, 0]]  # convert to rgb
        mergedMap = self.stageClassificationMap[:, :, [2, 1, 0]]

        cancerSmoothImg = cv2.GaussianBlur(cancerImg, (5, 5), 0)

        aspect = cancerImg.shape[0] / cancerImg.shape[1]

        width = 6.4 * 1
        mpl.rcParams["figure.figsize"] = [width * 1.10, width * aspect]
        mpl.rcParams["figure.dpi"] = 600
        #         print(aspect)

        if aspect > 1:
            hspace = 0.04
            wspace = hspace * aspect
        else:
            wspace = 0.04
            hspace = wspace / aspect  # / aspect
        # * aspect
        fig2, axarr = plt.subplots(2, 2, gridspec_kw={'wspace': 0.28, 'hspace': hspace})

        caxarr = []
        for r in range(2):
            for c in range(2):
                divider = make_axes_locatable(axarr[r, c])
                cax = divider.append_axes("right", size="5%", pad=0)
                caxarr.append(cax)
        caxarr = np.array(caxarr).reshape(2, 2)

        for x in [0, 1]:
            for y in [0, 1]:
                # axarr[x, y].axis('off')
                # axarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                # axarr[x, y].set_aspect(lymImg.shape[0] / lymImg.shape[1])

                # axarr[x, y].axis('off')
                axarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                caxarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                caxarr[x, y].axis("off")


        axarr[0, 0].imshow(thumbnail)

        axarr[1, 0].imshow(classificationMap)

        colors_classification = ['yellow', 'blue', 'gray']
        labels_classification = ['Tumor', 'Benign', 'Stroma']

        legend_patches_classification = [Rectangle((0, 0), width=0.01, height=0.01, color=icolor, label=label, lw=0)
                                         for icolor, label in zip(colors_classification, labels_classification)]

        caxarr[1, 0].legend(handles=legend_patches_classification,
                            facecolor=None,  # "white",
                            edgecolor=None,
                            fancybox=False,
                            bbox_to_anchor=(-0.1, 0),
                            loc='lower left',
                            fontsize='x-small',
                            shadow=False,
                            framealpha=0.,
                            borderpad=0)

        axarr[1, 1].imshow(mergedMap)

        colors_merge = ['#00FF00', 'orange', 'blue', 'gray']  # #00FF00 for pure green
        labels_merge = ['G3', 'G4+5', 'Benign', 'Stroma']

        legend_patches_merge = [Rectangle((0, 0), width=0.01, height=0.01, color=icolor, label=label, lw=0)
                                for icolor, label in zip(colors_merge, labels_merge)]

        caxarr[1, 1].legend(handles=legend_patches_merge,
                            facecolor=None,  # "white",
                            edgecolor="white",
                            fancybox=None,
                            bbox_to_anchor=(-0.1, 0),
                            loc='lower left',
                            fontsize='x-small',
                            borderpad=0)

        #         lymImg = classificationMap
        #         lymR = lymImg[:, :, 0]  # bgr? rgb
        #         lymInts = lymR.astype(np.float)
        #         lymInts = lymInts / 255

        #        lymB = lymImg[:, :, 2]  # b
        #        lymInts[lymB < 100] = None
        #        lymIm = axarr[1, 0].imshow(lymInts, cmap='jet', vmax=1.0, vmin=0.0)

        # heat map of cancer
        cancerR = cancerSmoothImg[:, :, 2]  # bgr
        cancerInts = cancerR.astype(np.float)
        cancerInts = cancerInts / 255

        cancerB = cancerSmoothImg[:, :, 0]  # b
        cancerInts[cancerB < 100] = None
        cancerIm = axarr[0, 1].imshow(cancerInts, cmap='jet', vmax=1.0, vmin=0.0)

        axins = inset_axes(caxarr[0, 1],
                           width="50%",  # width = 5% of parent_bbox width
                           height="60%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(0.2, 0, 1, 1),
                           bbox_transform=caxarr[0, 1].transAxes,
                           borderpad=0,
                           )

        cb = fig2.colorbar(cancerIm, cax=axins)
        cb.ax.tick_params(labelsize='xx-small')
        caxarr[0, 1].axis("off")

        #         plt.show()
        plt.savefig(self.savePath, bbox_inches='tight')
        plt.close()

    def saveImg_none_axes(self):
        shape = (self.cancerImg.shape[1] * 2, self.cancerImg.shape[0] * 2)
        thumbnail = self.oslide.get_thumbnail(shape)

        cancerImg = self.cancerImg

        classificationMap = self.classificationMap[:,:, [2, 1, 0]] # convert to rgb
        mergedMap = self.mergedMap[:,:, [2,1,0]]

        cancerSmoothImg = cv2.GaussianBlur(cancerImg, (5, 5), 0)

        aspect = cancerImg.shape[0] / cancerImg.shape[1]

        width = 6.4 * 1
        mpl.rcParams["figure.figsize"] = [width, width * aspect]
        mpl.rcParams["figure.dpi"] = 600
#         print(aspect)

        if aspect > 1:
            hspace = 0.04
            wspace = hspace * aspect
        else:
            wspace = 0.04
            hspace = wspace / aspect #/ aspect
         #* aspect
        fig2, axarr = plt.subplots(2, 2, gridspec_kw={'wspace':wspace, 'hspace': hspace})
        axarr[0, 0].imshow(thumbnail)



        axarr[1, 0].imshow(classificationMap)

        colors_classification = ['red', 'yellow', 'blue']
        labels_classification = ['Lymphocytes', 'Tumor', 'Tissue']

        legend_patches_classification = [Rectangle((0, 0), width=0.01, height=0.01, color=icolor, label=label, lw=0)
                                for icolor, label in zip(colors_classification, labels_classification)]

        axarr[1, 0].legend(handles=legend_patches_classification,
                           facecolor=None,  # "white",
                           edgecolor="white",
                           fancybox=None,
                           bbox_to_anchor=(1, 0),
                           loc='lower left',
                           borderpad=0)



        axarr[1, 1].imshow(mergedMap)

        colors_merge = ['red', 'yellow', 'blue']
        labels_merge = ['Lymphocytes', 'Tumor', 'Tissue']

        legend_patches_merge = [Rectangle((0, 0), width=0.01, height=0.01, color=icolor, label=label, lw=0)
                          for icolor, label in zip(colors_merge, labels_merge)]

        axarr[1, 1].legend(handles=legend_patches_merge,
                           facecolor=None,  # "white",
                           edgecolor="white",
                           fancybox=None,
                           bbox_to_anchor=(1, 0),
                           loc='lower left',
                           borderpad=0)

#         lymImg = classificationMap
#         lymR = lymImg[:, :, 0]  # bgr? rgb
#         lymInts = lymR.astype(np.float)
#         lymInts = lymInts / 255

#        lymB = lymImg[:, :, 2]  # b
#        lymInts[lymB < 100] = None
#        lymIm = axarr[1, 0].imshow(lymInts, cmap='jet', vmax=1.0, vmin=0.0)

        # heat map of cancer
        cancerR = cancerSmoothImg[:, :, 2]  # bgr
        cancerInts = cancerR.astype(np.float)
        cancerInts = cancerInts / 255

        cancerB = cancerSmoothImg[:, :, 0] # b
        cancerInts[cancerB < 100] = None
        cancerIm = axarr[0, 1].imshow(cancerInts, cmap='jet', vmax=1.0, vmin=0.0)

        axins = inset_axes(axarr[0, 1],
                           width="5%",  # width = 5% of parent_bbox width
                           height="50%",  # height : 50%
                           loc='lower left',
                           bbox_to_anchor=(1.05, 0., 1, 1),
                           bbox_transform=axarr[0, 1].transAxes,
                           borderpad=0,
                           )
        fig2.colorbar(cancerIm, cax=axins)

        for x in [0, 1]:
            for y in [0, 1]:
                # axarr[x, y].axis('off')
                axarr[x, y].tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
                # axarr[x, y].set_aspect(lymImg.shape[0] / lymImg.shape[1])


#         plt.show()
        plt.savefig(self.savePath)#, bbox_inches='tight')
        plt.close()


# these folders will be replaced by paramaters
svs_fol_staged = '/data10/shared/hanle/svs_SEER_PRAD'
staged_pred = '/data04/shared/hanle/prad_cancer_detection_SEER/data/heatmap_txt_3classes_header_seer1'
output_pred = '/data02/shared/hanle/test_4panedl_seer_prad'

# test_svs_id = 'TCGA-2A-A8VO-01Z-00-DX1'

prefix = "prediction-"
wsi_extension = ".tif"
skip_first_line_pred = True
args = None


def checkFileExisting(wsiId):
    allPath = [
        os.path.join(staged_pred, 'color-'+wsiId), # colorPath
        os.path.join(svs_fol_staged, wsiId+wsi_extension), # svsPath
        os.path.join(staged_pred, prefix+wsiId), # predPath
        #os.path.join(til_fol, 'prediction-'+wsiId), # tilPath_pred
        #os.path.join(til_fol, 'color-'+wsiId), # tilPath_color
    ]
    ans = True
    for path in allPath:
        ans = os.path.exists(path)
        if not ans:
            print(path, "does not exit!")
            break
    return ans


def gen1Image(f):
    wsiId = f[len(prefix):]
    if not checkFileExisting(wsiId):
        return
    oslide = openslide.OpenSlide(os.path.join(svs_fol_staged, wsiId+wsi_extension))
    stagedCancerFile = StagedTumorHeatMap(staged_pred, skip_first_line_pred)
    stagedCancerFile.setWidthHeightByOSlide(oslide)
    stagedCancerMap = stagedCancerFile.getHeatMapByID(wsiId)
    classificationMap = stagedCancerFile.getTumorClassificationMap()
    stageClassificationMap = stagedCancerFile.getStageClassificationMap()

    img = FourPanelGleasonImage(oslide, stagedCancerMap, classificationMap, stageClassificationMap,
                       os.path.join(output_pred, wsiId+".png"))
    img.saveImg()
    print(wsiId)


def main(parallel_processing = False):
    if not os.path.isdir(output_pred):
        os.makedirs(output_pred)
    print(prefix)

    grades_prediction_fns = [f for f in os.listdir(staged_pred) if f.startswith(prefix)]

    if len(grades_prediction_fns) == 0:
        print("No valid file!")
        return

    if not parallel_processing:
        for i, f in enumerate(grades_prediction_fns):
            gen1Image(f)
            print(i, f)
    else:
        num_of_cores = multiprocessing.cpu_count() - 2
        p = multiprocessing.Pool(num_of_cores)
        p.map(gen1Image, grades_prediction_fns)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

if __name__ == "__main__":
    is_parallel = True if str2bool(sys.argv[1]) else False
    main(parallel_processing = is_parallel)




