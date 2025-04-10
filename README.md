# Hyperspectral Remote Sensing Images Salient Object Detection: The First Benchmark Dataset and Baseline [TGRS 2025]
by [Peifu Liu](https://scholar.google.com/citations?user=yrRXe-8AAAAJ&hl=zh-CN), [Huiyan Bai](https://scholar.google.com/citations?user=0hhBs5AAAAAJ&hl=zh-CN),  [Tingfa Xu](https://scholar.google.com/citations?user=vmDc8dwAAAAJ&hl=zh-CN), Jihui Wang, [Huan Chen](https://scholar.google.com/citations?user=1G6Mj24AAAAJ&hl=zh-CN), and [Jianan Li](https://scholar.google.com.hk/citations?user=sQ_nP0ZaMn0C&hl=zh-CN&oi=ao).

[![arXiv](https://img.shields.io/badge/📃-arXiv-ff69b4)](https://arxiv.org/abs/2504.02416)


## Requirements
It is recommended to use Python 3.9. GDAL is available at [Google Drive](https://drive.google.com/file/d/1aMd_2w3khxUrur91F6RqFqeQD9RX1HXt/view?usp=drive_link). Feel free to use any PyTorch version.

## Dataset
The HRSSD is available at [Baidu Netdisk](https://pan.baidu.com/s/1xLHHTYerPYmKAFSYGC1TQg?pwd=jeng). Please place the dataset in the `dataset` folder. 
Our HRSSD is organized as follows:
```
/HRSSD
    /tr
        /image
        /label
        /mask
    /ts
        /image
        /label
        /mask
```

## Compared Methods
1. MJRBM: [Paper](https://ieeexplore.ieee.org/document/9511336), [Code](https://github.com/wchao1213/ORSI-SOD)  
2. FSMINet: [Paper](https://ieeexplore.ieee.org/document/9739705), [Code](https://github.com/zxforchid/FSMINet)  
3. CorrNet: [Paper](https://ieeexplore.ieee.org/document/9690514), [Code](https://github.com/MathLee/CorrNet)  
4. ACCoNet: [Paper](https://ieeexplore.ieee.org/document/9756652), [Code](https://github.com/MathLee/ACCoNet)  
5. SeaNet: [Paper](https://ieeexplore.ieee.org/document/9690514), [Code](https://github.com/MathLee/SeaNet)  
6. MEANet: [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0957417423022807), [Code](https://github.com/LiangBoCheng/MEANet)  
7. CTDNet: [Paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475494), [Code](https://github.com/zhaozhirui/CTDNet)  
8. TRACER: [Paper](https://ojs.aaai.org/index.php/AAAI/article/view/21633), [Code](https://github.com/Karel911/TRACER)  
9. BBRF: [Paper](https://ieeexplore.ieee.org/document/10006743), [Code](https://github.com/iCVTEAM/BBRF-TIP)  
10. MENet: [Paper](https://ieeexplore.ieee.org/document/10204274), [Code](https://github.com/yiwangtz/MENet)  
11. ADMNet: [Paper](https://ieeexplore.ieee.org/document/10555313), [Code](https://github.com/Kunye-Shen/ADMNet)  
12. SED/SG: [Paper](https://ieeexplore.ieee.org/document/6738493), [Code](https://github.com/liangjiecn/Saliency2013)  
13. SUDF: [Paper](https://www.sciencedirect.com/science/article/pii/S1570870520307125), [Code](https://github.com/gqding/SUDF)  
14. SMN: [Paper](https://ieeexplore.ieee.org/document/10313066), [Code](https://github.com/laprf/SMN)
15. CSCN: [Paper](https://ieeexplore.ieee.org/abstract/document/10613611), [Code](https://github.com/HuiyanBai/CSCN)
16. SAHRNet: [Paper](https://www.sciencedirect.com/science/article/pii/S092427162400025X), [Code](https://github.com/tulilin/Multitask_NCA)
17. MambaHSI: [Paper](https://ieeexplore.ieee.org/document/10604894), [Code](https://github.com/li-yapeng/MambaHSI)
18. MambaLG: [Paper](https://ieeexplore.ieee.org/document/10812905), [Code](https://github.com/danfenghong/IEEE_TGRS_MambaLG)
19. DSTC: [Paper](https://link.springer.com/chapter/10.1007/978-3-031-72754-2_21), [Code](https://github.com/laprf/DSTC)

## Acknowledgement
Our dataset is built upon [WHU-OHS](https://github.com/zjjerica/WHU-OHS-Pytorch). Thanks for their great work!

## License
This repository contains two components with different licenses:

Our **code** is released under the [![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE). You may freely use, modify, and distribute the code.

The **HRSSD dataset** is licensed under [![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC_BY--NC--ND_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/). It is intended for academic research only. You must attribute the original source, and you are not allowed to modify or redistribute the dataset without permission.