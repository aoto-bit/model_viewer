import tkinter as tk
from tkinter import Label, Frame, Button
from PIL import Image, ImageTk
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import json

# 画像の予測結果を表示するクラス
class ImagePredictor:
    def __init__(self, model, label_names):
        # 予測に使用するモデルとラベル名を設定
        self.model = model
        self.label_names = label_names

    def display_predictions(self, image, container):
        # 入力画像のチャンネル数を3に変換し、バッチ次元を追加
        image = image.repeat(3, 1, 1)
        image = image.unsqueeze(0)

        # モデルを使って予測を行い、上位9個の確率とラベルを取得
        with torch.no_grad():
            outputs = self.model(image)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        top_probs, top_labels = probs.topk(9)
        top_probs = top_probs.squeeze().tolist()
        top_labels = top_labels.squeeze().tolist()

        # 前の描画結果を消去
        for widget in container.winfo_children():
            widget.destroy()

        # 2つの円グラフを描画
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        labels = [self.label_names[label] for label in top_labels]
        ax1.pie(top_probs, labels=labels, autopct='%1.1f%%')
        ax1.set_title('AlexNet Predictions')

        # ランダムな確率分布を生成して2つ目の円グラフを描画
        randoms = sorted(np.random.dirichlet(np.ones(20), size=1)[0], reverse=True)[:9]
        labels = [f"No.{i}" for i in range(1,10)]
        ax2.pie(randoms, labels=labels, autopct='%1.1f%%')
        ax2.set_title('Random Distribution')

        # 円グラフをTkinterに埋め込む
        canvas = FigureCanvasTkAgg(fig, master=container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# 画像をロードして表示するクラス
class ImageLoader:
    def __init__(self, loader, display_func, container):
        # データローダーと表示関数、コンテナを設定
        self.loader = loader
        self.display_func = display_func
        self.container = container

    def load_images(self, frame_labels):
        # データローダーから6枚の画像を取得
        images, _ = next(iter(self.loader))
        for i in range(6):
            # 画像を前処理し、Tkinterで表示可能な形式に変換
            image = images[i].squeeze()
            pil_image = Image.fromarray((image.numpy() * 255).astype(np.uint8), 'L')
            tk_image = ImageTk.PhotoImage(pil_image)
            r, c = divmod(i, 3)
            label = frame_labels[r][c]
            label.config(image=tk_image)
            label.image = tk_image
            # 画像がクリックされたときの処理を設定
            label.bind("<Button-1>", lambda event, img=image: self.display_func(img, self.container))

# FashionMNISTビューアのメインクラス
class FashionMNISTViewer:
    def __init__(self, root):
        self.root = root
        self.root.title('FashionMNIST Image Viewer')
        self.main_frame = Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左側のフレームを作成
        self.left_frame = Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        # 右側のフレームを作成
        self.right_frame = Frame(self.main_frame, bg="gray", width=600, height=600)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # 左側のフレーム内にラベルを配置
        self.labels = [[Label(self.left_frame, borderwidth=2) for _ in range(3)] for _ in range(2)]
        for r in range(2):
            for c in range(3):
                self.labels[r][c].grid(row=r, column=c, padx=5, pady=5)

        # データとモデルを設定
        with open('../imagenet-simple-labels.json', 'r') as f:
            label_names = json.load(f)
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        model.eval()

        # 予測クラスとローダークラスを初期化
        self.image_predictor = ImagePredictor(model, label_names)
        self.image_loader = ImageLoader(test_loader, self.image_predictor.display_predictions, self.right_frame)

        # ボタンを配置
        self.reload_button = Button(self.left_frame, text="Reload Images", command=self.reload_images)
        self.reload_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.reload_images()  # 起動時に画像をロード

    def reload_images(self):
        # 画像をロードして表示
        self.image_loader.load_images(self.labels)

if __name__ == "__main__":
    # データ変換とローダーを設定
    image_width, image_height = 200, 200
    transform = transforms.Compose([
        transforms.Resize((image_width, image_height)),
        transforms.ToTensor()
    ])
    test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)

    root = tk.Tk()
    app = FashionMNISTViewer(root)
    root.mainloop()
