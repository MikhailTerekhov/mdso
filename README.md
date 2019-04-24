Direct Sparse Odometry for fisheye cameras
==========================================

This repository is an open-source implementation of [Direct Sparse Odometry](https://ieeexplore.ieee.org/abstract/document/7898369) algorithm generalized for fisheye cameras. [A paper](https://ieeexplore.ieee.org/abstract/document/8410468) from the authors of DSO exists, which describes how this generalization could be done. However, no open-source implementation is provided. We plan to expand our work even further to support arbitrary multi-camera systems with little to no FoV intersection.


Installing
----------

### Ubuntu
The following process was tested on clean Ubuntu 18.04

#### Prequisites
Firstly, you may need to install some required packages:
```bash
sudo apt install git
sudo apt install g++
sudo apt install cmake
sudo apt install libgflags-dev
sudo apt install libgoogle-glog-dev
sudo apt install libceres-dev
sudo apt install libtbb-dev
sudo apt install libopencv-dev
```
Addititonally, if you want to see nice graphs of errors, trajectories and more, you will need [Python 3](https://www.python.org/download/releases/3.0/) and [Matplotlib](https://matplotlib.org/):
```bash
sudo apt install python3
sudo apt install python3-matplotlib
```

#### Building
After that you can download and build the system with CMake (for best performance we recommend that you use RelWithDebInfo build configuration):
```bash
git clone https://bitbucket.org/slamgroup/dso/
cd dso
git submodule init
git submodule update
mkdir bin && cd bin
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make
```

Running provided demos
----------------------
Most of the demos support `--help` flag to show the detailed description of flags that could be used.
### [Multi-FoV](http://rpg.ifi.uzh.ch/fov.html) dataset
Fisheye Multi-FoV is currently the main dataset for testing purposes. You can currently run the odometry to generate trajectory and point cloud. It needs the full dataset to be present, including ground truth poses and depths. Expected structure of the data to be provided:
```
/path/to/MultiFoV
├── data
|   ├── img
|   |   ├── img0001_0.png
|   |   └── ...
|   └── depth
|       ├── img0001_0.depth
|       └── ...
└── info
    ├── depthmaps.txt
    ├── groundtruth.txt
    └── ... 
```

#### Run the odometry on Multi-FoV
To get the trajectory and point clouds you need the `genply` demo. It requires that empty folders for general output, debug images and tracking residuals are present. The default ones are `output/default`, `output/default/debug` and `output/default/track` respectively. By default `genply` runs on the whole dataset (which takes lots of time!), so you may want to run only on its segment. For this you may use `--start` and `--count` options. Considering you are in the root of the repository and you have already built the system, all you need to do is
```bash
mkdir -p output/default/debug output/default/track
./bin/samples/mfov/genply/genply /path/to/MultiFoV
```
Among the other stuff it generates `output/points.ply` point cloud, which you can inspect, for example, with the [MeshLab](http://www.meshlab.net/) tool. 

If you want to inspect the trajectory that is generated, you can do it with
```bash
python3 py/showtrack.py path/to/output/dir
```

### Other demos

* `triang` is a simple demo that shows Delaunay Triangulation (which is used in the initialization part of our system) of a random selection of points in the square. It can be run as simple as this:
```bash
./triang
```
* `selectpix` demonstrates the adaptive pixel selection algorithm on a video of your choice. It could be run with
```bash
./selectpix dir
```
where dir names a directory with video frames stored as jpg or png files. Frames should be ordered alphabetically.

Built With
----------

* [CMake](https://cmake.org/) - Crossplatform build system
* [Eigen](http://eigen.tuxfamily.org/) - Template library for linear algebra
* [Sophus](https://github.com/strasdat/Sophus) - Template implementation of geometrical Lie groups (SO(3), SE(3), etc.)
* [GFlags](https://github.com/gflags/gflags) - Command-line flag parsing library
* [GLog](https://github.com/google/glog) - Logging library
* [ceres-solver](http://ceres-solver.org/) - Scalable library for solving optimization problems
* [TBB](https://www.threadingbuildingblocks.org/) - Framework for parallelization 
* [OpenCV](https://opencv.org/) - Open-source Computer Vision library

License
----------
This project is licensed under the terms of the MIT license. For more information, please check the `LICENSE.txt` file.
