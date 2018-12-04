# System
Anaconda3 in ubuntu 18.04

## Env Setup
After Anaconda3 for Pyhon 3.7 is installed (default)
~~~sh
conda create --name env python=3.7
~~~
If python is version 3.6
~~~sh
conda create --name env python=3.6 anaconda
~~~
Then activate
~~~sh
source activate env
~~~
Due to some potential version problems, plz first install tensorflow and related package
~~~sh
conda install tensorflow
conda install matplotlib
pip install opencv-contrib-python
pip install facenet
pip install dlib
~~~

# Main Dependent package
## installed
~~~sh
opencv numpy facenet tensorflow scipy matplotlib os dlib
~~~

# Expression Labels
~~~sh
0 = neutral
1 = angry
2 = contempt
3 = disgust
4 = fear 
5 = happy
6 = sadness
7 = surprise
~~~