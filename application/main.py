# -*- coding: utf-8 -*-
import sys
import warnings
from PyQt5.QtWidgets import (
    QApplication,
    QLabel,
    QWidget,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QFileDialog,
)
from PyQt5.QtCore import Qt

import torch
import torchvision.transforms as transforms
from vgg import Vgg
from loader import get_dataloaders, custom_imageloader, get_dataloaders2, custom_imageloader2
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import torchvision.models as models
import resnet34

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setGeometry(200, 200, 400, 400)
        self.setWindowTitle("fer-2013-application")

        # Setup UI
        button = [None] * 5
        button[0] = QPushButton("Load model", self)
        button[0].clicked.connect(self.load_model)
        self.model_label = QLabel("No model selected.")
        self.model_label.setAlignment(Qt.AlignCenter)
        button[1] = QPushButton("Test random dataset image", self)
        button[1].clicked.connect(self.test_random_image)
        button[2] = QPushButton("Test image", self)
        button[2].clicked.connect(self.test_image)
        button[3] = QPushButton("Real-time camera detect", self)
        button[3].clicked.connect(self.camera_detect)

        vbox = QVBoxLayout()
        vbox.addWidget(button[0])
        vbox.addWidget(self.model_label)
        vbox.addWidget(button[1])
        vbox.addWidget(button[2])
        vbox.addWidget(button[3])
        vbox.addStretch(1)
        self.setLayout(vbox)

        self.net = None
        print("loading data...")
        (   self.trainloader,
            self.valloader,
            self.testloader,
            self.emotion_mapping,
            self.emotion_mapping2,
        ) = get_dataloaders(bs=1)
        
        (   self.trainloader2,
            self.valloader2,
            self.testloader2,
        ) = get_dataloaders2(bs=1)

    # Button 0
    def load_model(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
        else:
            return
        filename = filenames[0]
        self.model_label.setText("Model: " + filename)
        print(filename)
        # Load model
        name_split = filename.split('/')
        if name_split[-1] == 'MYVGG':
            self.net_mode = 0
            checkpoint = torch.load(filename)
            self.net = Vgg().to(device)
            self.net.load_state_dict(checkpoint["params"])
            self.net.eval()
        else:
            self.net_mode = 1
            self.net = torch.load(filename)
            self.net.to(device)
            self.net.eval()

    # Button 1
    def test_random_image(self):
        # get image from trainloader
        if self.net_mode ==0:
            
            image, label = next(iter(self.testloader))
        else:
            image, label = next(iter(self.testloader2))
        image_plt = torch.squeeze(image)

        # activate gradients for input image
        image = image.to(device)
        image.requires_grad_()

        # get prediction and score
        output = self.net(image)
        prediction = output.argmax()
        output_max = output[0, prediction]

        # backprop score
        output_max.backward()

        # calculate saliency, rescale between 0 and 1
        saliency, _ = torch.max(image.grad.data.abs(), dim=1)
        if self.net_mode == 0:
            saliency = saliency.reshape(40, 40)
        else:
            saliency = saliency.reshape(48, 48)
        saliency -= saliency.min(1, keepdim=True)[0]
        saliency /= saliency.max(1, keepdim=True)[0]

        # draw picture
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        plt.tight_layout()

        label_str = self.emotion_mapping[label.item()]
        axs[0].set_title("Emotion: " + label_str, fontsize=18)
        axs[0].imshow(image_plt, cmap="gray")
        axs[0].axis("off")

        axs[1].set_title("Saliency Map", fontsize=18)
        im = axs[1].imshow(saliency.cpu(), cmap="Blues")
        axs[1].axis("off")
        if self.net_mode == 0:
            pred_str = self.emotion_mapping[prediction.item()]
        else:
            pred_str = self.emotion_mapping2[prediction.item()]
        axs[2].set_title("Predict: " + pred_str, fontsize=18)
        axs[2].imshow(image_plt, cmap="gray")
        axs[2].imshow(saliency.cpu(), cmap="Blues", alpha=0.4)
        axs[2].axis("off")

        cbar = fig.colorbar(im, ax=axs.ravel().tolist())
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(12)

        plt.savefig("saliency.png", bbox_inches="tight")
        plt.show()

    # Button 2
    def test_image(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
        else:
            return
        filename = filenames[0]
        
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            haar_img = cv2.rectangle(img.copy(), (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            crop_gray = gray[y:y + h, x:x + w]
            dataloader = custom_imageloader(crop_gray)
            image, _ = next(iter(dataloader))
            image_plt = torch.squeeze(image)

            # activate gradients for input image
            image = image.to(device)
            image.requires_grad_()

            # get prediction and score
            output = self.net(image)
            prediction = output.argmax()
            output_max = output[0, prediction]

            # backprop score
            output_max.backward()

            # calculate saliency, rescale between 0 and 1
            saliency, _ = torch.max(image.grad.data.abs(), dim=1)
            saliency = saliency.reshape(40, 40)
            saliency -= saliency.min(1, keepdim=True)[0]
            saliency /= saliency.max(1, keepdim=True)[0]

            # draw picture
            fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15, 5))
            plt.tight_layout()

            axs[0].set_title("Detect Face", fontsize=18)
            axs[0].imshow(cv2.cvtColor(haar_img, cv2.COLOR_BGR2RGB))
            axs[0].axis("off")

            axs[1].set_title("Crop Image", fontsize=18)
            axs[1].imshow(image_plt, cmap="gray")
            axs[1].axis("off")

            axs[2].set_title("Saliency Map", fontsize=18)
            im = axs[2].imshow(saliency.cpu(), cmap="Blues")
            axs[2].axis("off")

            pred_str = self.emotion_mapping[prediction.item()]
            axs[3].set_title("Predict: " + pred_str, fontsize=18)
            axs[3].imshow(image_plt, cmap="gray")
            axs[3].imshow(saliency.cpu(), cmap="Blues", alpha=0.4)
            axs[3].axis("off")

            cbar = fig.colorbar(im, ax=axs.ravel().tolist())
            for t in cbar.ax.get_yticklabels():
                t.set_fontsize(12)

            plt.savefig("saliency.png", bbox_inches="tight")
            plt.show()
    
    # Button 3
    def camera_detect(self):
        # prevents openCL usage and unnecessary logging messages
        cv2.ocl.setUseOpenCL(False)
        
        # start the webcam feed
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                crop_gray = gray[y:y + h, x:x + w]
                if self.net_mode == 0:
                    dataloader = custom_imageloader(crop_gray)
                else:
                    dataloader = custom_imageloader2(crop_gray)
                image, _ = next(iter(dataloader))

                # activate gradients for input image
                image = image.to(device)
                image.requires_grad_()

                # get prediction and score
                output = self.net(image)
                prediction = output.argmax()
                if self.net_mode==0:    
                    pred_str = self.emotion_mapping[prediction.item()]
                else:
                    pred_str = self.emotion_mapping2[prediction.item()]
                cv2.putText(frame, pred_str, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            cv2.imshow('Video', cv2.resize(frame, (1600,960), interpolation = cv2.INTER_CUBIC))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
