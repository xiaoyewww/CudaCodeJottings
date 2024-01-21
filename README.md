# CudaCodeJottings

Cuda code Jottings what have learnt.

## Build

### Linux

```
cmake -S . -B build
cmake --build build --config release -j4
```

## Contents

```
├── include
├── src
├── test
├── CMakeLists.txt
└── README.md
```

### Others

Todo...

## Test

```
cd build
ctest
```

- optional:
    - `-R <ut>`: choose ut
    - `-V`: print console output of unittests

## Referece

- [cuda-samples](https://github.com/NVIDIA/cuda-samples/tree/master)
