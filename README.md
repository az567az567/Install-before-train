# Install and create environment
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\extras\demo_suite  

@Install tensorflow @  
------

第一步就是安装Anaconda  
https://ithelp.ithome.com.tw/articles/10229662?sc=pt  
安装cuda 11.2和cudnn 8.1.0.77  
在 Windows 上從原始碼開始建構  |  TensorFlow (google.cn)  
CUDA Toolkit 11.5 Downloads | NVIDIA Developer  
cuDNN Download | NVIDIA Developer  
環境變數  
Cuda和Cudnn安装_liuzhonglin_的博客-CSDN博客_cuda cudnn  
創建一個名為tensorflow-gpu且python3.9的虛擬環境  
安装CUDA安装cuDNN  
安装tensorflow的GPU版本  
Tensorflow-gpu安装超详细！！！_m0_49090516的博客-CSDN博客_tensorflowgpu安装  
參考   
Win 10安裝TensorFlow GPU並在Jupyter Notebook和Spyder運行 | by Rick | Medium  
[機器學習 ML NOTE] Windows 搭建Tensorflow-GPU 環境(Anaconda + Tensorflow-GPU+ CUDA+ cuDNN) | by GGWithRabitLIFE | 雞雞與兔兔的工程世界 | Medium  
Cuda11.2, cudnn8.1, tf=2.6  
https://blog.csdn.net/qq_42388742/article/details/111245578  
study  
https://tf.wiki/zh_hans/basic/tools.html#tf-config-gpu  
https://www.youtube.com/watch?v=tpCFfeUEGs8&list=PL6vjgQ2-qJFfU2vF6-lG9DlSa4tROkzt9&index=1&ab_channel=DanielBourke  
conda create -n tensorflow-gpu pip python=3.8  

conda下載問題
------  

https://stackoverflow.com/questions/64589421/packagesnotfounderror-cudatoolkit-11-1-0-when-installing-pytorch  
activate tensorflow-gpu  
conda install cudatoolkit=11.2 -c conda-forge  
conda install cudnn=8.1 -c conda-forge   
pip install tensorflow-gpu==2.6.0  


conda activate tf1  
conda create --name py37 python=3.7 -c conda-forge  
conda create -n tf1 pip python=3.7 -c conda-forge  
conda install --use-local C:\Users\az567\Downloads\cudatoolkit-10.0.130-0.conda  
conda install --use-local C:\Users\az567\Downloads\cudnn-7.6.0-cuda10.0_0.conda  
pip install tensorflow-gpu==1.15  
python  
import tensorflow as tf  
tf.__version__  
tf.test.is_gpu_available()  
