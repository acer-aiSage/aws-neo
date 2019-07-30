sudo apt-get update
sudo apt-get install python3-pip python3-setuptools cython3 python3-pil python3-numpy python3-h5py build-essential manpages-dev python3-dev python3-psutil git
sudo apt-get install -f ./acer-opencv3.deb
git clone https://github.com/neo-ai/neo-ai-dlr.git
cd neo-ai-dlr/
git checkout -b demo origin/demo-aisage
cd demo/aisage/
sudo pip3 install ./tensorflow-1.13.1-cp35-none-linux_aarch64.whl
sudo pip3 install --upgrade dlr-1.0-py2.py3-none-any.whl
sudo easy_install3 tvm-0.6.dev0-py3.5-linux-aarch64.egg
