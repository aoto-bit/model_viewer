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

# display_predictions関数
def display_predictions(image):
    # チャンネル数を1から3に変換
    image = image.repeat(3, 1, 1)

    # モデルへの入力準備(バッチ次元を追加)
    image = image.unsqueeze(0)  

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

    # 円グラフを2つ作成するためのサブプロットを設定
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))  

    # ラベル名のリストを作成
    labels = [label_names[label] for label in top_labels]
    
    # 1つ目の円グラフを描画
    ax1.pie(top_probs, labels=labels, autopct='%1.1f%%')
    ax1.axis('equal')  
    ax1.set_title('AlexNet predictive labels inferred by fashion-minist')
    
    # ランダムな確率分布を生成
    randoms = sorted(np.random.dirichlet(np.ones(20), size=1)[0], reverse=True)[:9]

    # 2つ目の円グラフを描画
    labels = [f"No.{i}" for i in range(1,10)]
    ax2.pie(randoms, labels=labels, autopct='%1.1f%%')
    ax2.axis('equal')
    ax2.set_title('probability distribution')

    # 円グラフをTkinterに埋め込む
    canvas = FigureCanvasTkAgg(fig, master=right_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

# on_image_click関数
def on_image_click(event, r, c):
    # グローバル変数を使用
    global selected_label
    # すべてのラベルの枠をリセット
    for row in labels:
        for label in row:
            label.config(relief="groove")
    # 選択されたラベルの枠を強調
    event.widget.config(relief="raised", borderwidth=8)
    # 選択された画像の座標を保存
    selected_label = (r, c)
    # 画像選択ボタンを有効化
    button_select.config(state=tk.NORMAL)

# select_image関数
def select_image():
    # 画像が選択されている場合
    if selected_label:
        # 選択された画像の座標を取得
        r, c = selected_label
        # 選択された画像データを取得
        image = test_dataset[c * 2 + r][0]  
        # 選択された画像の予測結果を表示
        display_predictions(image)

# load_images関数
def load_images():
    # 画像選択ボタンを無効化
    button_select.config(state=tk.DISABLED)
    # データローダーから6枚の画像を取得
    images, _ = next(iter(test_loader))
    # 6枚の画像を表示
    for i in range(6):
        # 画像データを処理
        image = images[i].squeeze()
        image = (image.numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image, 'L')
        tk_image = ImageTk.PhotoImage(pil_image)
        # 画像を表示するラベルの座標を計算
        r, c = divmod(i, 3)
        label = labels[r][c]
        # ラベルに画像を設定
        label.config(image=tk_image)
        label.image = tk_image  # 画像がガベージコレクションされないように参照を保持
        # クリックイベントを設定
        label.bind("<Button-1>", lambda event, r=r, c=c: on_image_click(event, r, c))

# 画像をリロードするボタン
button_reload = Button(frame_buttons, text="Reload Images", command=load_images)
button_reload.pack(side=tk.LEFT, padx=5)

# 画像を選択するボタン
button_select = Button(frame_buttons, text="Select Image", command=select_image, state=tk.DISABLED)
button_select.pack(side=tk.LEFT, padx=5)

load_images()  # 初期画像をロード
root.mainloop()
