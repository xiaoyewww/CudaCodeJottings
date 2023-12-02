# CudaCodeJottings

Cuda code Jottings what have learnt.

## Build

### Linux

```
cmake -S . -B build
cmake --build build --config release -j4
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
