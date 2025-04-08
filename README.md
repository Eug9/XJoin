### XJoin: Portable, parallel hash join across diverse XPU architectures with oneAPI

XJoin implements the hash join database operator. 
It is the DPC++ implementation of the hash join operator originally written in CUDA.


##### Compile

```
mkdir build
cd build
cmake ..
make
```

##### Run

```
./join size_of_build_table
```

### Publication

**XJoin: Portable, parallel hash join across diverse XPU architectures with oneAPI** <br>
_DAMON '21: Proceedings of the 17th International Workshop on Data Management on New Hardware, Article No.: 11, Pages 1 - 5,
https://doi.org/10.1145/3465998.3466012_
