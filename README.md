# mutas

### AR-app

**Linux环境下，python版本及主要第三方库依赖：**

python：3.6

pip3 install opencv-python 安装opencv可能遇到的问题：https://stackoverflow.com/questions/47113029/importerror-libsm-so-6-cannot-open-shared-object-file-no-such-file-or-directo

pip3 install dlib 安装dlib可能遇到的问题：https://blog.csdn.net/zhangjunbob/article/details/73431076

pip3 install hyperlpr

pip3 install Pillow

pip3 install numpy

**Linux环境下，AR-app运行示例：**

python3 main4face-detect.py input_video/baby.mp4

python3 main4hat-generate.py input_video/baby.mp4

python3 main4license-plate-recognize.py input_video/car.mp4
