<br/>
<p align="center">
  <h3 align="center">DeepLabV3-Segmentation</h3>

  <p align="center">
    A DeepLab V3+ Model with ResNet 50 / ResNet101 Encoder to perform Binary Segmentation Tasks. 
    <br/>
    <br/>
    <a href="https://github.com/mukund-ks/DeepLabV3-Segmentation/issues">Report Bug</a>
    .
    <a href="https://github.com/mukund-ks/DeepLabV3-Segmentation/issues">Request Feature</a>
  </p>
</p>

![Downloads](https://img.shields.io/github/downloads/mukund-ks/DeepLabV3-Segmentation/total) ![Contributors](https://img.shields.io/github/contributors/mukund-ks/DeepLabV3-Segmentation?color=dark-green) ![Forks](https://img.shields.io/github/forks/mukund-ks/DeepLabV3-Segmentation?style=social) ![Stargazers](https://img.shields.io/github/stars/mukund-ks/DeepLabV3-Segmentation?style=social) ![Issues](https://img.shields.io/github/issues/mukund-ks/DeepLabV3-Segmentation) ![License](https://img.shields.io/github/license/mukund-ks/DeepLabV3-Segmentation) 

## Table Of Contents

- [Table Of Contents](#table-of-contents)
- [About The Project](#about-the-project)
- [Results](#results)
  - [EWS](#ews)
- [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
  - [Creating A Pull Request](#creating-a-pull-request)
- [License](#license)
- [Authors](#authors)
- [Acknowledgements](#acknowledgements)
- [To Cite this Repository](#to-cite-this-repository)

## About The Project

The goal of this research is to develop a DeepLabV3+ model with a choice of ResNet50 or ResNet101 backbone to perform binary segmentation on plant image datasets. Based on the presence or absence of a certain object or characteristic, binary segmentation entails splitting an image into discrete subgroups known as image segments which helps to simplify processing or analysis of the image by reducing the complexity of the image. Labeling pixels is another step in the segmentation process. Each pixel or piece of a picture assigned to the same category has a unique label. 


Plant pictures with ground truth binary mask labels make up the training and validation dataset. The project uses Tensorflow, a well-known deep learning library, for model development, training, and evaluation.[^1] During the training process, the model is optimized using strategies like the Dice Loss, Adam optimizer, Reducing LR on Pleateau and Early Stopping. All the while, important metrics like Intersection over Union (IoU), Precision, Recall, Accuracy and Dice Coefficient are kept track of.

[^1]: A PyTorch implementation can be found [here](https://github.com/mukund-ks/DeepLabV3Plus-PyTorch). 

_Datasets used during development of this project are given below:_
* [EWS Dataset](https://www.research-collection.ethz.ch/handle/20.500.11850/512332)

* [Plant Semantic Segmentation Dataset by HIL](https://humansintheloop.org/resources/datasets/plant-segmentation/)

* [CVPPP Dataset](https://www.plant-phenotyping.org/datasets-home)

The ultimate objective of the project is to develop a strong model that can accurately segment plant-related regions inside photographs, which can have applications in a variety of fields, such as agriculture, botany, and environmental sciences. The included code demonstrates how to prepare the data, create the model's architecture, train it on the dataset, and assess the model's effectiveness using a variety of metrics.

## Results

Results of the developed Model on EWS, PSS and CVPPP Dataset.

### EWS 

On the basis of IoU, the results of this repository are compared to [Zenkl et al. (2022)](https://www.frontiersin.org/articles/10.3389/fpls.2021.774068/full), [Yu et al. (2017)](https://plantmethods.biomedcentral.com/articles/10.1186/s13007-017-0168-4), [Sadeghi-Tehran et al. (2020)](https://spj.science.org/doi/10.34133/2020/3521852) and [Rico-Fernández et al. (2018)](https://www.sciencedirect.com/science/article/abs/pii/S0168169918301911?via%3Dihub).

A Development Flowchart and configurations of different model versions for the EWS Dataset can be accessed [here](https://t.ly/eYQHe).


| Benchmark                        |    IoU    |
| :------------------------------- | :-------: |
| **Repository (Model V1.5)**      | **0.768** |
| **Zenkl et al. (2022)**          |   0.775   |
| **Yu et al. (2017)**             |   0.666   |
| **Sadeghi-Tehran et al. (2020)** |   0.638   |
| **Rico-Fernández et al. (2018)** |   0.691   |


## Built With
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)


![Tensorflow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)


* IDE Used:


![VSCode](https://img.shields.io/badge/VSCode-0078D4?style=for-the-badge&logo=visual%20studio%20code&logoColor=white)

* Operating System:

![Windows11](https://img.shields.io/badge/Windows_11-0078d4?style=for-the-badge&logo=windows-11&logoColor=white)

## Getting Started

To get a local copy of this project up and running on your machine, follow these simple steps.

* Clone a copy of this Repository on your machine.
```console
git clone https://github.com/mukund-ks/DeepLabV3-Segmentation.git
```

### Prerequisites

* Python 3.9 or above.

```console
python -V
Python 3.9.13
```

* CUDA 11.2 or above.

```console
nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Sun_Feb_14_22:08:44_Pacific_Standard_Time_2021
Cuda compilation tools, release 11.2, V11.2.152
Build cuda_11.2.r11.2/compiler.29618528_0
```

### Installation

1. Move into the cloned repo.
```console
cd DeepLabV3-Segmentation
```

2. Setup a Virutal Environment

```console
python -m venv env
```

3. Activate the Virutal Environment
```console
env/Scripts/activate
```

4. Install Dependencies

```console
pip install -r requirements.txt
```

> **Note**
> You can deactivate the Virtual Environment by using
> ```env/Scripts/deactivate```
 

## Usage

The Model can be trained on the data aforementioned in the [**About**](#about-the-project) section or on your own data.

* To train & evaluate the model, use [`main.py`](https://github.com/mukund-ks/DeepLabV3-Segmentation/blob/main/main.py)
```console
python main.py --help
```
```console
Usage: main.py [OPTIONS]

  A DeepLab V3+ Decoder based Binary Segmentation Model with choice of
  Encoders b/w ResNet101 and ResNet50.

  Please make sure that your data is structured according to the folder
  structure specified in the Github Repository.

  See: https://github.com/mukund-ks/DeepLabV3-Segmentation

Options:
  --data-dir TEXT                 Path for Data Directory.  [required]
  --eval-dir TEXT                 Path for Evaluation Directory.  [required]
  -m, --model-type [ResNet101|ResNet50]
                                  Choice of Encoder.  [required]
  -a, --augmentation BOOLEAN      Opt-in to apply augmentations to provided
                                  data. Default - True
  -s, --split-data BOOLEAN        Opt-in to split data into Training and
                                  Validation set. Default - True
  --stop-early BOOLEAN            Opt-in to stop Training early if val_loss
                                  isn't improving. Default - True
  -b, --batch-size INTEGER        Batch size of data during training. Default
                                  - 4
  -e, --epochs INTEGER            Number of epochs during training. Default -
                                  25
  --help                          Show this message and exit.
```

## Folder Structure

The folder structure will alter slightly depending on whether or not your training data has already been divided into a training and testing set.

- If the data is not already seperated, it should be in a directory called `data` that is further subdivided into `Image` and `Mask` subdirectories.
  
  - [`main.py`](https://github.com/mukund-ks/DeepLabV3-Segmentation/blob/main/main.py) should be run with `--split-data` option as `True` in this case.
  
    Example: ```python main.py --data-dir data --eval-dir eval_data --model-type ResNet50 --split-data True```

> **Note**
> The data will be split into training and testing set with a ratio of 0.2

```console
$ tree -L 2
.
├── data
│   ├── Image
│   └── Mask
└── eval_data
    ├── Image
    └── Mask
```

- If the data has already been separated, it should be in a directory called `data` that is further subdivided into the subdirectories `Train` and `Test`, both of which contain the subdirectories `Image` and `Mask`.

  - [`main.py`](https://github.com/mukund-ks/DeepLabV3-Segmentation/blob/main/main.py) should be run with `--split-data` option as `False` in this case.
  
    Example: ```python main.py --data-dir data --eval-dir eval_data --model-type ResNet50 --split-data False```

```console
$ tree -L 3
.
├── data
│   ├── Test
│   │   ├── Image
│   │   └── Mask
│   └── Train
│       ├── Image
│       └── Mask
└── eval_data
    ├── Image
    └── Mask
```
* The structure of `eval_data` remains the same in both cases, holding `Image` and `Mask` sub-directories.

> **Note**
> The directory names are case-sensitive.
## Roadmap

See the [open issues](https://github.com/mukund-ks/DeepLabV3-Segmentation/issues) for a list of proposed features (and known issues).

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/mukund-ks/DeepLabV3-Segmentation/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/mukund-ks/DeepLabV3-Segmentation/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b MyBranch`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push -u origin myBranch`)
5. Open a Pull Request

## License

Distributed under the Apache 2.0 License. See [LICENSE](https://github.com/mukund-ks/DeepLabV3-Segmentation/blob/main/LICENSE) for more information.

## Authors

* [**Mukund Kumar Surehli**](https://github.com/mukund-ks/) - *Comp Sci Student* 
* [**Naveen Aggarwal**](https://github.com/navagg) - *Comp Sci Professor* - *Project Guide* 

## Acknowledgements

* M. Minervini, A. Fischbach, H.Scharr, and S.A. Tsaftaris. [_Finely-grained annotated datasets for image-based plant phenotyping._](https://www.sciencedirect.com/science/article/abs/pii/S0167865515003645?via%3Dihub) Pattern Recognition Letters, pages 1-10, 2015, [doi:10.1016/j.patrec.2015.10.013](https://www.sciencedirect.com/science/article/abs/pii/S0167865515003645?via%3Dihub)
* H. Scharr, M. Minervini, A.P. French, C. Klukas, D. Kramer, Xiaoming Liu, I. Luengo, J.-M. Pape, G. Polder, D. Vukadinovic, Xi Yin, and S.A. Tsaftaris. [_Leaf segmentation in plant phenotyping: A collation study._](https://link.springer.com/article/10.1007/s00138-015-0737-3) Machine Vision and Applications, pages 1-18, 2015, [doi:10.1007/s00138-015-0737-3.](https://link.springer.com/article/10.1007/s00138-015-0737-3)
* B. Dellen, H. Scharr, and C. Torras. [_Growth signatures of rosette plants from time-lapse video._](https://ieeexplore.ieee.org/document/7044561) IEEE/ACM Transactions on Computational Biology and Bioinformatics, PP(99):1 - 11, 2015, [doi:10.1109/TCBB.2015.2404810](https://ieeexplore.ieee.org/document/7044561)
* E.E. Aksoy, A. Abramov, F. Wörgötter, H. Scharr, A. Fischbach, and B. Dellen. [_Modeling leaf growth of rosette plants using infrared stereo image sequences._](https://www.sciencedirect.com/science/article/abs/pii/S0168169914002816?via%3Dihub) Computers and Electronics in Agriculture, 110:78 - 90, 2015, [doi:10.1016/j.compag.2014.10.020](https://www.sciencedirect.com/science/article/abs/pii/S0168169914002816?via%3Dihub)
* M. Minervini , M.M. Abdelsamea, S.A. Tsaftaris. [_Image-based plant phenotyping with incremental learning and active contours._](https://www.sciencedirect.com/science/article/abs/pii/S1574954113000691?via%3Dihub) Ecological Informatics 23, 35–48, 2014, [doi:10.1016/j.ecoinf.2013.07.004](https://www.sciencedirect.com/science/article/abs/pii/S1574954113000691?via%3Dihub)
* Polat H. [_A modified DeepLabV3+ based semantic segmentation of chest computed tomography images for COVID-19 lung infections._](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9349869/) Int J Imaging Syst Technol. 2022;32(5):1481-1495. [doi:10.1002/ima.22772](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9349869/)
* Li, K. (2022). [_Study on the segmentation method of the improved deeplabv3+ algorithm in the basketball scene._](https://www.hindawi.com/journals/sp/2022/3311931/) Scientific Programming, 2022, 1–7. https://doi.org/10.1155/2022/3311931
* Wang Y, Wang C, Wu H, Chen P (2022) [_An improved Deeplabv3+ semantic segmentation algorithm with multiple loss constraints._](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0261582) PLOS ONE 17(1): e0261582. https://doi.org/10.1371/journal.pone.0261582
* Zenkl, R., Timofte, R., Kirchgessner, N., Roth, L., Hund, A., Van Gool, L., Walter, A., &amp; Aasen, H. (2022). [_Outdoor plant segmentation with deep learning for high-throughput field phenotyping on a diverse wheat dataset._](https://www.frontiersin.org/articles/10.3389/fpls.2021.774068/full) Frontiers in Plant Science, 12. https://doi.org/10.3389/fpls.2021.774068 
* Hsu C-Y, Hu R, Xiang Y, Long X, Li Z. [_Improving the Deeplabv3+ Model with Attention Mechanisms Applied to Eye Detection and Segmentation._](https://www.mdpi.com/2227-7390/10/15/2597) Mathematics. 2022; 10(15):2597. https://doi.org/10.3390/math10152597
* Singh, V. (2023, January 17). _The Ultimate Guide to deeplabv3 - with Pytorch Inference._ LearnOpenCV. https://learnopencv.com/deeplabv3-ultimate-guide/ 
* Zualkernan, I., Abuhani, D. A., Hussain, M. H., Khan, J., & ElMohandes, M. (2023). [_Machine Learning for Precision Agriculture Using Imagery from Unmanned Aerial Vehicles (UAVs): A Survey._](https://www.mdpi.com/2504-446X/7/6/382) Drones, 7(6), 382. https://doi.org/10.3390/drones7060382
* S. Minaee, Y. Boykov, F. Porikli, A. Plaza, N. Kehtarnavaz and D. Terzopoulos, [_Image Segmentation Using Deep Learning: A Survey_](https://ieeexplore.ieee.org/document/9356353), in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 7, pp. 3523-3542, 1 July 2022, [doi: 10.1109/TPAMI.2021.3059968](https://ieeexplore.ieee.org/document/9356353).
* Pröve, P. L. (2017, October 18). _Squeeze-and-Excitation Networks_. Retrieved from https://towardsdatascience.com/squeeze-and-excitation-networks-9ef5e71eacd7
* Chen, LC., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). [_Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation_](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_49). In: Ferrari, V., Hebert, M., Sminchisescu, C., Weiss, Y. (eds) Computer Vision – ECCV 2018. ECCV 2018. Lecture Notes in Computer Science(), vol 11211. Springer, Cham. https://doi.org/10.1007/978-3-030-01234-2_49
* Zhou, E., Xu, X., Xu, B. et al. [_An enhancement model based on dense atrous and inception convolution for image semantic segmentation_](https://link.springer.com/article/10.1007/s10489-022-03448-w). Appl Intell 53, 5519–5531 (2023). https://doi.org/10.1007/s10489-022-03448-w
* M. S. Minhas, _Transfer Learning for Semantic Segmentation using PyTorch DeepLab v3_, GitHub.com/msminhas93, 12-Sep-2019. Available: https://github.com/msminhas93/DeepLabv3FineTuning.
* Kou, L., Sysyn, M., Fischer, S., Liu, J., & Nabochenko, O. (2022). [_Optical Rail Surface Crack Detection Method Based on Semantic Segmentation Replacement for Magnetic Particle Inspection_](https://www.mdpi.com/1424-8220/22/21/8214). Sensors, 22(21), 8214. https://doi.org/10.3390/s22218214
* Zhang C, Gao S, Yang X, Li F, Yue M, Han Y, Zhao H, Zhang Y, Fan K. [_Convolutional Neural Network-Based Remote Sensing Images Segmentation Method for Extracting Winter Wheat Spatial Distribution_](https://www.mdpi.com/2076-3417/8/10/1981). Applied Sciences. 2018; 8(10):1981. https://doi.org/10.3390/app8101981
* Zhang, D., Zhang, L. & Tang, J. [_Augmented FCN: rethinking context modeling for semantic segmentation_](https://link.springer.com/article/10.1007/s11432-021-3590-1). Sci. China Inf. Sci. 66, 142105 (2023). https://doi.org/10.1007/s11432-021-3590-1
* Zeiler, M. D., & Fergus, R. (2013, November 12). _Visualizing and Understanding Convolutional Networks_. Retrieved from https://arxiv.org/abs/1311.2901v3
* Chen, L., Papandreou, G., Schroff, F., & Adam, H. (2017). [_Rethinking Atrous Convolution for Semantic Image Segmentation_](https://www.semanticscholar.org/paper/Rethinking-Atrous-Convolution-for-Semantic-Image-Chen-Papandreou/ee4a012a4b12d11d7ab8c0e79c61e807927a163c). ArXiv, abs/1706.05587.
* Z. Zhang, X. Wang and C. Jung, [_DCSR: Dilated Convolutions for Single Image Super-Resolution_](https://ieeexplore.ieee.org/document/8502129), in IEEE Transactions on Image Processing, vol. 28, no. 4, pp. 1625-1635, April 2019, [doi: 10.1109/TIP.2018.2877483](https://ieeexplore.ieee.org/document/8502129).
* [EWS Dataset](https://doi.org/10.3389/fpls.2021.774068)
* [Plant Phenotyping Dataset](https://www.plant-phenotyping.org/datasets)
* [Plant Semantic Segmentation Dataset by HIL](https://humansintheloop.org/resources/datasets/plant-segmentation/)


## To Cite this Repository


Surehli, M. K., & Aggarwal, N. (2023, August 7). GitHub - mukund-ks/DeepLabV3-Segmentation: _A DeepLab V3+ Model with ResNet 50 / ResNet101 Encoder for Binary Segmentation. Implemented with Tensorflow._ Retrieved from https://github.com/mukund-ks/DeepLabV3-Segmentation
