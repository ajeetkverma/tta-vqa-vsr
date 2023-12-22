# steps to create the environment---

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

Download the networks.zip fle and copy correlation_cpp ad correlation_cuda
cd networks/correlation_cpp
python setup.py install

cd networks/correlation_cuda
python setup.py install

# comment out the devices which are not workig fine.