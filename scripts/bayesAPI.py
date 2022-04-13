import multiprocessing
import cv2
import time
import sys
from scripts.bayesmat import main
from PIL import Image
import numpy as np
from threading import Thread
import os
from scripts.getKnownRegions import getKnownRegions

class bayesMat:
    def __init__(self, **arguments):
        """
        The Function that takes keyword arguments for bayesMat class.
        """
        full_path = os.path.realpath(__file__)
        self.path, filename = os.path.split(full_path)
        self.results = []
        self.core = arguments["cpu_count"]
        self.input = arguments["input"]
        self.trimap = arguments["trimap"]
        self.N = arguments["N"]
        self.sigma_c = arguments["sigma_c"]
        self.sigma_g = arguments["sigma_g"]
        self.minN = arguments["minN"]
        self.max_iter = arguments["max_iter"]
        self.min_like = arguments["min_like"]
        if arguments["output_name"] is not None:
            if arguments["output_name"].count('.') == 0:
                self.output_name = arguments["output_name"] + ".png"
            else:
                self.output_name = arguments["output_name"]
        else:
            self.output_name = "output.png"
        if arguments["progress_bar"] is not None:
            self.progressFlag = arguments["progress_bar"]
        else:
            self.progressFlag = False

    def getResult(self, data):
        """
        The function that acquires results from multiprocess functions.
        """
        self.results.append(data)

    def splitImage(self, img, file_name):
        """
        The function that splits the images according to core count.
        """
        img = Image.open(img)
        self.imgSize = img.size
        width, height = self.imgSize
        
        w, h = int(np.ceil(width / 2)), int(np.ceil(height / (self.core // 2)))
        frame_num = 0
        for col_i in range(0, width, w):
            for row_i in range(0, height, h):
                crop = img.crop((col_i, row_i, col_i + w, row_i + h))
                crop.save(
                        self.path + "/data/" + file_name + f"_{frame_num}.png")
                frame_num += 1

    def combineImg(self):
        """
        The function that combines output images.
        """
        new_im = Image.new('RGB', (self.imgSize[0], self.imgSize[1]))
        file_name = self.path + "/data/output{0}.png" 
        output = []

        w = h = 0
        for i in range(self.core // 2):
            im = Image.open(file_name.format(i))
            new_im.paste(im,(w,h))
            h += im.size[1]
            output.append(file_name.format(i))

        w += im.size[0]
        h = 0
        for i in range(self.core // 2, self.core):
            im = Image.open(file_name.format(i))
            new_im.paste(im,(w,h))
            h += im.size[1]
            output.append(file_name.format(i))

        new_im.save(self.output_name)
        new_im = np.array(new_im)
        new_im[new_im > 155] = 255
        new_im[new_im != 255] = 0
        new_im = new_im * 255
        new_im = cv2.cvtColor(new_im, cv2.COLOR_RGB2BGR)
        self.clearImgs(output)
        return new_im

    def readImgs(self):
        """
        The function that reads splitted images for process.
        """
        inputs = []
        trimaps = []
        for i in range(self.core):
            inputs.append(self.path + "/data/input_{0}.png".format(i))
            trimaps.append(self.path + "/data/trimap_{0}.png".format(i))

        return inputs, trimaps

    def clearImgs(self, imgs):
        """
        The function that clears images after process is done.
        """
        import os
        for img in imgs:
            os.remove(img)

    def progressBar(
            self,
            iteration,
            total,
            prefix='',
            suffix='',
            decimals=1,
            length=100,
            fill='█',
            printEnd="\r"
            ):
        percent = ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
        if iteration == total:
            print()

    def showProgressBar(self):
        self.progressBar(
            0,
            self.core,
            prefix="Progress:", suffix="Complete", length=100)

        old_len = 0
        while True:
            curr_len = len(self.results)
            if curr_len != old_len:
                self.progressBar(
                    curr_len,
                    self.core,
                    prefix="Progress:",
                    suffix="Complete", length=100)
            
            if len(self.results) == self.core:
                break

    def run(self):
        self.splitImage(self.input, "input")
        self.splitImage(self.trimap, "trimap")

        inputs, trimaps = self.readImgs()
        start = time.time()
        pool = multiprocessing.Pool(processes=self.core)
        for n in range(len(inputs)):
            try:
                pool.apply_async(main, args=[
                                            inputs[n],
                                            trimaps[n],
                                            self.N,
                                            self.sigma_c,
                                            self.sigma_g,
                                            self.minN,
                                            self.max_iter,
                                            self.min_like,
                                            False
                                            ], callback=self.getResult)
            except:
                print("Something wrong")

        if self.progressFlag:
            Thread(target=self.showProgressBar, args=()).start()

        pool.close()
        pool.join()
        calc_time = int(time.time() - start)

        for i in range(len(self.results)):
            cv2.imwrite(
                self.path + "/data/output{0}.png".format(self.results[i][0]),
                self.results[i][1] * 255)

        self.clearImgs(inputs)
        self.clearImgs(trimaps)
        alpha = self.combineImg()
        img = cv2.imread(self.input)
        cv2.imwrite("composited_" + self.output_name, np.multiply(img, alpha)) 
        return alpha, calc_time
