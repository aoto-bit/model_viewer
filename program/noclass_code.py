import tkinter as tk
from tkinter import Label, Frame, Button
from PIL import Image, ImageTk
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import torch

import random

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import json
from translate import Translator

# translator = Translator(from_lang = "en", to_lang = "ja")
# mpl.rcParams['font.family'] = 'Google Noto'

with open('/root/feature_maps/fashon_MNIST/imagenet-simple-labels.json', 'r') as f:
    label_names = json.load(f)

# 画像の表示サイズを設定
image_width, image_height = 200, 200

# データの変換処理
transform = transforms.Compose([
    transforms.Resize((image_width, image_height)),
    transforms.ToTensor()
])

# データセットとデータローダーの準備
test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)

# AlexNetモデルの準備
model = models.alexnet(pretrained=True)
model.eval()

def display_predictions(image):
    # チャンネル数を1から3に変換
    image = image.repeat(3, 1, 1)

    # モデルへの入力準備
    image = image.unsqueeze(0)  # バッチ次元を追加

    # 確率を計算
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)

    # 上位9個のラベルとその確率を取得
    top_probs, top_labels = probs.topk(9)
    top_probs = top_probs.squeeze().tolist()
    top_labels = top_labels.squeeze().tolist()

    # 右側のフレームを初期化
    for widget in right_frame.winfo_children():
        widget.destroy()

    # 円グラフを2つ作成
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  # 1行2列のサブプロット

    labels = [label_names[label] for label in top_labels]
    
    # 1つ目の円グラフ
    ax1.pie(top_probs, labels=labels, autopct='%1.1f%%')
    ax1.axis('equal')  # 円グラフを正円にするために必要
    ax1.set_title('AlexNet predictive labels inferred by fashion-minist')
    
    randoms = sorted(np.random.dirichlet(np.ones(20), size=1)[0], reverse=True)[:9]

    # 2つ目の円グラフ
    labels = [f"No.{i}" for i in range(1,10)]
    ax2.pie(randoms, labels=labels, autopct='%1.1f%%')
    ax2.axis('equal')
    ax2.set_title('probability distribution')

    # 円グラフをTkinterに埋め込む
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# Tkinter GUIを設定
root = tk.Tk()
root.title('FashionMNIST Image Viewer')

# メインフレームを作成
main_frame = Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

# 左側のフレーム (画像表示用)
left_frame = Frame(main_frame)
left_frame.pack(side=tk.LEFT, padx=10, pady=10)

# 画像表示用のフレーム
frame_images = Frame(left_frame, relief="groove", highlightbackground="red", highlightcolor="red")
frame_images.pack(padx=10, pady=10)

# ボタン用のフレーム
frame_buttons = Frame(left_frame)
frame_buttons.pack(padx=10, pady=10)

# 右側のフレーム (大きな領域)
right_frame = Frame(main_frame, bg="gray", width=600, height=600)
right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

# 画像表示用のラベルと枠のデフォルトスタイルを生成
labels = [[Label(frame_images, borderwidth=2) for _ in range(3)] for _ in range(2)]
for r in range(2):
    for c in range(3):
        labels[r][c].grid(row=r, column=c, padx=5, pady=5)

selected_label = None

def on_image_click(event, r, c):
    """ 画像がクリックされた時の処理 """
    global selected_label
    # すべてのラベルの枠をリセット
    for row in labels:
        for label in row:
            label.config(relief="groove")
    # 選択されたラベルの枠を強調
    event.widget.config(relief="raised", borderwidth=8)
    selected_label = (r, c)
    button_select.config(state=tk.NORMAL)

def select_image():
    """ 選択された画像に対する操作 """
    if selected_label:
        r, c = selected_label
        image = test_dataset[c * 2 + r][0]  # 選択された画像を取得
        display_predictions(image)

def load_images():
    button_select.config(state=tk.DISABLED)
    """FashionMNISTデータセットから6枚の画像をロードして表示する"""
    images, _ = next(iter(test_loader))
    for i in range(6):
        image = images[i].squeeze()
        image = (image.numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image, 'L')
        tk_image = ImageTk.PhotoImage(pil_image)
        r, c = divmod(i, 3)
        label = labels[r][c]
        label.config(image=tk_image)
        label.image = tk_image  # 画像がガベージコレクションにより消えないように参照を保持
        label.bind("<Button-1>", lambda event, r=r, c=c: on_image_click(event, r, c))

# 画像をリロードするボタン
button_reload = Button(frame_buttons, text="Reload Images", command=load_images)
button_reload.pack(side=tk.LEFT, padx=5)

# 画像を選択するボタン
button_select = Button(frame_buttons, text="Select Image", command=select_image, state=tk.DISABLED)
button_select.pack(side=tk.LEFT, padx=5)

load_images()  # 初期画像をロード
root.mainloop()