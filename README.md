# 環境構築
- 必要な物
    - docker desktop
    - GPU
- 手順
    - Dockerの起動まで  
        - docker pull pytorch/pytorch
        - docker images (imageがあるか確認)
        - docker run -it ID
    - Dockerの起動後
        - apt-get update
        - apt-get install -y tk-dev libx11-dev
        - apt install fontconfig

