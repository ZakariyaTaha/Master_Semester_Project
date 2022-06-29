# Master_Semester_Project

## Please check the report to know my contribution, as well as mainResUnet which I added afterwards.

In order to get the best results, the ones I achieved after the final report, you need first to download the pre-trained model of MedicalNet at  [Google Drive](https://drive.google.com/file/d/13tnSvXY7oDIEloNFiGTsjUIYfS3g3BfG/view?usp=sharing) from 

```
    @article{chen2019med3d,
        title={Med3D: Transfer Learning for 3D Medical Image Analysis},
        author={Chen, Sihong and Ma, Kai and Zheng, Yefeng},
        journal={arXiv preprint arXiv:1904.00625},
        year={2019}
    }
```

After that adapt the dataset in topoloss4neurons/datasets/myDataset.py to match your directories (you need to have created the binary rendered files and distance maps beforehand). Finally, you can run using this command after adapting the mainResUnet.config file to your data:

````
python mainResUnet.py -c mainResUnet.config 
