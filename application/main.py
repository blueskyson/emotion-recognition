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
from loader import get_dataloaders, custom_imageloader
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np

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

        vbox = QVBoxLayout()
        vbox.addWidget(button[0])
        vbox.addWidget(self.model_label)
        vbox.addWidget(button[1])
        vbox.addWidget(button[2])
        vbox.addStretch(1)
        self.setLayout(vbox)

        self.net = None
        print("loading data...")
        (
            self.trainloader,
            self.valloader,
            self.testloader,
            self.emotion_mapping,
        ) = get_dataloaders(bs=1)

    # button 0
    def load_model(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            filenames = dlg.selectedFiles()
        else:
            return
        filename = filenames[0]
        self.model_label.setText("Model: " + filename)

        # Load model
        checkpoint = torch.load(filename)
        self.net = Vgg().to(device)
        self.net.load_state_dict(checkpoint["params"])
        self.net.eval()

    def test_random_image(self):
        # get image from trainloader
        image, label = next(iter(self.testloader))
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
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
        plt.tight_layout()

        label_str = self.emotion_mapping[label.item()]
        axs[0].set_title("Emotion: " + label_str, fontsize=18)
        axs[0].imshow(image_plt, cmap="gray")
        axs[0].axis("off")

        axs[1].set_title("Saliency Map", fontsize=18)
        im = axs[1].imshow(saliency.cpu(), cmap="Blues")
        axs[1].axis("off")

        pred_str = self.emotion_mapping[prediction.item()]
        axs[2].set_title("Predict: " + pred_str, fontsize=18)
        axs[2].imshow(image_plt, cmap="gray")
        axs[2].imshow(saliency.cpu(), cmap="Blues", alpha=0.4)
        axs[2].axis("off")

        cbar = fig.colorbar(im, ax=axs.ravel().tolist())
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(12)

        plt.savefig("saliency.png", bbox_inches="tight")
        plt.show()

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
            cv2.rectangle(img, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            crop_gray = gray[y:y + h, x:x + w]
            pil = Image.fromarray(crop_gray)
            image = custom_imageloader(pil)
            # image_plt = torch.squeeze(image)

            # # activate gradients for input image
            # image = image.to(device)
            # image.requires_grad_()

            # # get prediction and score
            # output = self.net(image)
            # prediction = output.argmax()
            # output_max = output[0, prediction]

            # # backprop score
            # output_max.backward()

            # # calculate saliency, rescale between 0 and 1
            # saliency, _ = torch.max(image.grad.data.abs(), dim=1)
            # saliency = saliency.reshape(40, 40)
            # saliency -= saliency.min(1, keepdim=True)[0]
            # saliency /= saliency.max(1, keepdim=True)[0]

            # # draw picture
            # fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
            # plt.tight_layout()

            # axs[0].set_title("Image", fontsize=18)
            # axs[0].imshow(image_plt, cmap="gray")
            # axs[0].axis("off")

            # axs[1].set_title("Saliency Map", fontsize=18)
            # im = axs[1].imshow(saliency.cpu(), cmap="Blues")
            # axs[1].axis("off")

            # pred_str = self.emotion_mapping[prediction.item()]
            # axs[2].set_title("Predict: " + pred_str, fontsize=18)
            # axs[2].imshow(image_plt, cmap="gray")
            # axs[2].imshow(saliency.cpu(), cmap="Blues", alpha=0.4)
            # axs[2].axis("off")

            # cbar = fig.colorbar(im, ax=axs.ravel().tolist())
            # for t in cbar.ax.get_yticklabels():
            #     t.set_fontsize(12)

            # plt.savefig("saliency.png", bbox_inches="tight")
            # plt.show()   


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
