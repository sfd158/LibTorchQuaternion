@echo off
git pull
cd CudaRotation
rmdir /S /Q build
rmdir /S /Q out
del *.pyd *.so
pip install -e .
cd ..\CppRotation
rmdir /S /Q build
rmdir /S /Q out
del *.pyd *.so
pip install -e .
cd ..\TorchRotation
rmdir /S /Q build
rmdir /S /Q out
del *.pyd *.so
pip install -e .
cd ..
