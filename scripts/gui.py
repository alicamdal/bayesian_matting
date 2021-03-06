from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, \
    QFileDialog, QLabel, QLineEdit, QProgressBar, QDesktopWidget, \
    QSizePolicy, QCheckBox
from PyQt5 import QtCore
import sys
from PyQt5.QtCore import pyqtSlot, QThread
from scripts.bayesAPI import bayesMat
from scripts.bayesmat import main
import time
from multiprocessing.pool import ThreadPool
import multiprocessing
import numpy as np
import cv2
import math
from PIL import Image

class progressBar(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Progress')
        self.setGeometry(400, 400, 400, 60)
        self.status_label = QLabel("Processing", self)
        self.status_label.move(150, 30)
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.core = multiprocessing.cpu_count()
        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 400, 25)
        self.progress.setMaximum(100)
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def updateProgress(self, bayes, cpu):
        old_len = 0
        self.progress.setValue(0)
        self.status_label.setText("Processing")
        while True:
            curr_len = len(bayes.results)
            if curr_len != old_len:
                old_len = curr_len
                self.progress.setValue(np.ceil(curr_len * (100 / cpu)))
            if curr_len == cpu:
                break
            QApplication.processEvents()
        self.status_label.setText("Completed")
        QApplication.processEvents()
        time.sleep(2)
        self.close()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.second = progressBar()
        self.setGeometry(400, 400, 800, 400)
        self.setWindowTitle("Team 3D")

        self.inpButton = QPushButton(self)
        self.inpButton.setText("Select Input")
        self.inpButton.clicked.connect(self.inpDialog)
        self.inpButton.move(250, 50)
        self.startButton = QPushButton(self)
        self.startButton.setText("Start")
        self.startButton.clicked.connect(self.start)
        self.startButton.move(0, 300)
        self.triButton = QPushButton(self)
        self.triButton.setText("Select Trimap")
        self.triButton.clicked.connect(self.triDialog)
        self.triButton.move(250, 100)
        self.triButton = QPushButton(self)
        self.triButton.setText("Select GT")
        self.triButton.clicked.connect(self.gtDialog)
        self.triButton.move(250, 150)
        self.bgButton = QPushButton(self)
        self.bgButton.setText("Select BG")
        self.bgButton.clicked.connect(self.bgDialog)
        self.bgButton.move(250, 200)

        self.inplbl = QLabel("Input Image : ", self)
        self.inplbl.move(0, 50)
        self.inp_name = QLabel("None", self)
        self.inp_name.move(120, 50)
        self.trilbl = QLabel("Trimap Image : ", self)
        self.trilbl.move(0, 100)
        self.tri_name = QLabel("None", self)
        self.tri_name.move(120, 100)
        self.gtLbl = QLabel("GT Image : ", self)
        self.gtLbl.move(0, 150)
        self.gt_name = QLabel("None", self)
        self.gt_name.move(120, 150)
        self.bgLbl = QLabel("BG Image : ", self)
        self.bgLbl.move(0, 200)
        self.bg_name = QLabel("None", self)
        self.bg_name.move(120, 200)
        
        self.mseLbl = QLabel("MSE : ", self)
        self.mseLbl.move(0, 250)
        self.psnrLbl = QLabel("PSNR : ", self)
        self.psnrLbl.move(120, 250)
        self.timeLbl = QLabel("Time : ", self)
        self.timeLbl.move(240, 250)

        self.time_value = QLabel("None", self)
        self.time_value.move(290, 250)
        self.mse_value = QLabel("None", self)
        self.mse_value.move(50, 250)
        self.psnr_value = QLabel("None", self)
        self.psnr_value.move(170, 250)

        self.windowsizelbl = QLabel("Window Size : ", self)
        self.windowsizelbl.move(450, 50)
        self.sigmaclbl = QLabel("Sigma C : ", self)
        self.sigmaclbl.move(450, 100)
        self.sigmaglbl = QLabel("Sigma G : ", self)
        self.sigmaglbl.move(450, 150)
        self.minNlbl = QLabel("Min Number : ", self)
        self.minNlbl.move(450, 200)
        self.maxiterlbl = QLabel("Max iteration : ", self)
        self.maxiterlbl.move(450, 250)
        self.minlikelbl = QLabel("Min Likelihood : ", self)
        self.minlikelbl.move(450, 300)
        self.cpu_count = QLabel("CPU Count : ", self)
        self.cpu_count.move(450, 350)

        self.windowSize = QLineEdit(self)
        self.windowSize.move(600, 50)
        self.windowSize.insert("25")
        self.sigmaC = QLineEdit(self)
        self.sigmaC.move(600, 100)
        self.sigmaC.insert("0.01")
        self.sigmaG = QLineEdit(self)
        self.sigmaG.move(600, 150)
        self.sigmaG.insert("10")
        self.minN = QLineEdit(self)
        self.minN.move(600, 200)
        self.minN.insert("40")
        self.maxiter = QLineEdit(self)
        self.maxiter.move(600, 250)
        self.maxiter.insert("200")
        self.minlike = QLineEdit(self)
        self.minlike.move(600, 300)
        self.minlike.insert("1e-6")
        self.get_cpu_count = QLineEdit(self)
        self.get_cpu_count.move(600, 350)
        self.get_cpu_count.insert(str(multiprocessing.cpu_count()))

        self.show()

    @pyqtSlot()
    def inpDialog(self):
        self.inp_file, check = QFileDialog.getOpenFileName(
                                                            None,
                                                            "Select Image",
                                                            "",
                                                            "All Files (*);;"
                                                            )
        if check:
            name = self.inp_file.split("/")
            self.inp_name.setText(name[-1])

    @pyqtSlot()
    def triDialog(self):
        self.trimap_file, check = QFileDialog.getOpenFileName(
                                                            None,
                                                            "Select Image",
                                                            "",
                                                            "All Files (*);;"
                                                            )
        if check:
            name = self.trimap_file.split("/")
            self.tri_name.setText(name[-1])

    @pyqtSlot()
    def gtDialog(self):
        self.gt_file, check = QFileDialog.getOpenFileName(
                                                            None,
                                                            "Select Image",
                                                            "",
                                                            "All Files (*);;"
                                                            )
        if check:
            name = self.gt_file.split("/")
            self.gt_name.setText(name[-1])

    @pyqtSlot()
    def bgDialog(self):
        self.bg_file, check = QFileDialog.getOpenFileName(
                                                            None,
                                                            "Select Image",
                                                            "",
                                                            "All Files (*);;"
                                                            )
        if check:
            name = self.bg_file.split("/")
            self.bg_name.setText(name[-1])

    def calculateMetrics(self, alpha, calc_time, selector):
        if selector == 0:
            gt_img = cv2.imread(self.gt_file) / 255
            self.mse = np.mean((gt_img - alpha) ** 2)
            self.psnr = (
                        20 * math.log10(1.0 / math.sqrt(self.mse))
                        )
            psnr = "%.2f" % self.psnr
            mse = "%.5f" % self.mse
            time = "%.2f" % calc_time
            self.psnr_value.setText(psnr)
            self.mse_value.setText(mse)
            self.time_value.setText(time)
        else:
            gt_img = cv2.imread(self.gt_file, 0) / 255
            self.mse = np.mean((gt_img - alpha) ** 2)
            self.psnr = (
                        20 * math.log10(1.0 / math.sqrt(self.mse))
                        )
            psnr = "%.2f" % self.psnr
            mse = "%.5f" % self.mse
            time = "%.2f" % calc_time
            self.psnr_value.setText(psnr)
            self.mse_value.setText(mse)
            self.time_value.setText(time)

    def createComposite(self, bg_inp, fg_inp, alpha):
        new_bg = cv2.imread(bg_inp) / 255
        fg_inp = cv2.imread(fg_inp) / 255
        alpha_cp = alpha
        if new_bg.shape[0] != fg_inp.shape[0] or new_bg.shape[1] != fg_inp.shape[1]:
            new_shp_bg = Image.new("RGB", (fg_inp.shape[1], fg_inp.shape[0]))
            new_shp_bg.paste(Image.open(bg_inp), (0, 0))
            new_bg = np.array(new_shp_bg)
            new_bg = cv2.cvtColor(new_bg, cv2.COLOR_RGB2BGR)
            new_bg = new_bg / 255
            print("""
                ERROR
                New Background width or height is different than input's.
                """)

        if alpha_cp.shape[-1] != 3:
            new_alpha = Image.new("RGB", (alpha_cp.shape[1], alpha_cp.shape[0]))
            new_alpha.paste(Image.fromarray(alpha_cp * 255), (0,0))
            alpha_cp = np.array(new_alpha)
            #alpha_cp[alpha_cp > 155] = 255
            #alpha_cp[alpha_cp != 255] = 0
            alpha_cp = cv2.cvtColor(alpha_cp, cv2.COLOR_RGB2BGR) / 255

        alpha_mask = alpha_cp.copy()
        alpha_mask[alpha_cp != 0] = 0
        alpha_mask[alpha_cp == 0] = 1
        bg = alpha_mask * new_bg
        fg = alpha_cp * fg_inp
        result = np.hstack((alpha, bg + fg))
        cv2.imshow("results", result)
        cv2.imwrite("outputs/output_alpha.png", alpha * 255)
        cv2.imwrite("outputs/output_foreground.png", fg * 255)
        cv2.imwrite("outputs/output_background.png", bg * 255)
        cv2.imwrite("outputs/output_composited.png", (bg + fg) * 255)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            cv2.destroyAllWindows()

    @pyqtSlot()
    def start(self):
        cpu = int(self.get_cpu_count.text())
        if cpu != 1:
            inputf = self.inp_file
            trimap = self.trimap_file
            N = int(self.windowSize.text())
            sigma_c = float(self.sigmaC.text())
            sigma_g = float(self.sigmaG.text())
            minN = int(self.minN.text())
            max_iter = int(self.maxiter.text())
            min_like = float(self.minlike.text())

            self.bayes = bayesMat(
                    input=inputf,
                    trimap=trimap,
                    N=N,
                    sigma_c=sigma_c,
                    sigma_g=sigma_g,
                    minN=minN,
                    max_iter=max_iter,
                    min_like=min_like,
                    progress_bar=False,
                    output_name=None,
                    cpu_count=cpu
                )
            result = ThreadPool(processes=1).apply_async(self.bayes.run)
            self.second.show()
            self.second.updateProgress(self.bayes, cpu)
            alpha, calc_time = result.get()
            self.calculateMetrics(alpha, calc_time=calc_time, selector=0)
        else:
            inputf = self.inp_file
            trimap = self.trimap_file
            N = int(self.windowSize.text())
            sigma_c = float(self.sigmaC.text())
            sigma_g = float(self.sigmaG.text())
            minN = int(self.minN.text())
            max_iter = int(self.maxiter.text())
            min_like = float(self.minlike.text())
            start = time.time()
            alpha = main(
                        inputf,
                        trimap,
                        N,
                        sigma_c,
                        sigma_g,
                        minN,
                        max_iter,
                        min_like,
                        True
                        )
            calc_time = time.time() - start
            self.calculateMetrics(alpha, calc_time=calc_time, selector=1)

        self.createComposite(self.bg_file, self.inp_file, alpha)
