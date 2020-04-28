# SPLINE-Net
SPLINE-Net: Sparse Photometric Stereo through Lighting Interpolation and Normal Estimation Networks
http://openaccess.thecvf.com/content_ICCV_2019/papers/Zheng_SPLINE-Net_Sparse_Photometric_Stereo_Through_Lighting_Interpolation_and_Normal_Estimation_ICCV_2019_paper.pdf <br>

## Dependencies

- Python 3.5+
- PyTorch 0.4.0+
- TensorFlow 1.3+

## Training 

- Will upload soon

## Test SPLINE-Net on DiLiGenT Dataset

```shell
# Prepare the test set which consists of 100 subsets, 10 lightings each object
sh prepare_diligent_testset.sh
# This command will download and unzip the test set

# Download pre-trained model
sh download_pretrained_model.sh

# Run for test
python main.py --mode test
# Please check the results in photometric/results
```

## Test SPLINE-Net on your own dataset
- Please follow the data format of test set we created, use 'data/test' as a reference.
- In text file 'data/test/.../..txt', each line is a 1 * 24 vector, elem1, ..., elem24, represent data of one pixel. 
- Elem1 is index of the pixel in the original image. Elem2, elem3, elem4 are nx, ny, nz in normal vector.
- Elem5, ..., elem14 are index of the observation map, mapping to 10 lightings.
- Elem15, ..., elem24 are corresponding intensities.

## Citation
If you find our code is useful, please cite our paper. If you have any problem of implementation or running the code, please contact us: <br>csqianzheng@gmail.com<br>, <br>jiaym15@outlook.com<br> 
```
@inproceedings{zheng2019spline,
  title={SPLINE-Net: Sparse photometric stereo through lighting interpolation and normal estimation networks},
  author={Zheng, Qian and Jia, Yiming and Shi, Boxin and Jiang, Xudong and Duan, Ling-Yu and Kot, Alex C},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={8549--8558},
  year={2019}
}
```