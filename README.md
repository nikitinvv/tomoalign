# tomoalign

## 1. create conda environment and install dependencies

```console
conda create -n tomoalign -c conda-forge cupy swig scikit-build dxchange opencv tomopy
```

Note: CUDA drivers need to be installed before installation

## 2. install

```console
python setup.py install
```

## 3. check adjoint tests

```console
cd tests

```

Test deformation

```console
python test_deform.py

```
sample output:

```console
registration time: 5.561098337173462
apply flow time: 0.01982426643371582
data0-data1=5928.0068359375
data0-data1_unwrap=2606.488037109375
norm flow = 10330.0009765625
<data,D*Ddata>=<Ddata,Ddata>: 1.108183e+08 ? 1.164992e+08
```

Test tomography

```console
python test_tomo.py
```

sample output:

```console
norm data = 21722.1015625
norm object = 4057758.5
<u,R*Ru>=<Ru,Ru>: 4.718072e+08+0.000000e+00j ? 4.718676e+08
```

## 4. experimental data reconstruction

battery reconstruction:

```console
cd experimental/battery
```

processing.py - preprocessing

test_center.py - find rotation center

cg.py - convetional reconstruction by CG

admm.py - admm-based reconstruction with optical flow alignment
