git pull
cd CudaRotation
rm -rf build out *.pyd *.so
pip install -e .

cd ../CppRotation
rm -rf build out *.pyd *.so
pip install -e .

cd ../TorchRotation
rm -rf build out *.pyd *.so
pip install -e .

cd ..
