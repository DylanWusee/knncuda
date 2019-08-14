nvcc src/knncuda.cu -o knncuda.cu.o -c -lcuda -lcublas -lcudart -Wno-deprecated-gpu-targets -x cu -Xcompiler -fPIC
g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` src/knncuda.cpp knncuda.cu.o -o knncuda.so -I /usr/local/cuda/include -lcudart -L /usr/local/cuda/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -lcublas

