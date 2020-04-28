# SPLINE-Net
SPLINE-Net: Sparse Photometric Stereo through Lighting Interpolation and Normal Estimation Networks
https://arxiv.org/abs/1905.04088 <br>

## Dependencies

- Python 3.5+
- PyTorch 0.4.0+
- TensorFlow 1.3+

## Training 

Will upload soon

## Test SPLINE-Net on DiLiGenT Dataset

```shell
# Prepare the test set which consists of 100 subsets, 10 lightings each object
sh prepare_diligent_testset.sh
# This command will download and unzip the test set

# Run for test
python main.py --mode test
# Please check the results in photometric/results
```

## Test SPLINE-Net on your own dataset
Please follow the data format of test set we created, use 'data/test' as a reference.
In text file 'data/test/.../..txt', each line is a 1 * 24 vector, elem1, ..., elem24, represent data of one pixel. 
Elem1 is index of the pixel in the original image. Elem2, elem3, elem4 are nx, ny, nz in normal vector.
Elem5, ..., elem14 are index of the observation map, mapping to 10 lightings.
Elem15, ..., elem24 are corresponding intensities   