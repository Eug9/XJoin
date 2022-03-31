# XJoin

XJoin implement the hash join database operator. 
It is the DPC++ implementation of the hash join operator originally written in CUDA.

## Usage

### On a Linux Machine

First compile the sources
```bash
mkdir build
cd build
cmake ..
make
```
then run the executable specifying the size of the table

```bash
./join <size_of_build_table>
```

### On Intel DevCloud

Create a script `launch.sh` like the following one

```bash
#!/bin/sh
echo "Starting ..."
cd build
cmake ..
make
./join <size_of_build_table>
echo "Finished!"
```

then submit a job, for instance to a GPU worker

```bash
qsub -l nodes=1:gpu:ppn=2 -d . launch.sh
```

Eventually, see the output in a `launch.sh.oxxxxxxx` and the (hopefully empty) error trace in `launch.sh.exxxxxxx`.

**Notes:** 
* `-l nodes=1:gpu:ppn=2` is used to assign one full GPU node to the job.
* `-d .` is used to configure the current folder as the working directory for the task.
* `launch.sh` is the script that gets executed on the compute node.
* to run on a CPU, just run `qsub -d . launch.sh`



