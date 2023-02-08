# scMSI
scMSI: accurately estimating the Micro-Satellite Instability on sub-clones by a comprehensive deconvolution model on length spectrum
--

scMSI adopts an alternating iterative model to de-convolute the length distribution, which is a mixture of sub-clones. The two optimization stages are coordinated and gradually alternated to eliminate the mixed microsatellite length distribution in tumors and quantify the the microsatellite length distribution parameters. Finally, a joint deconvolution on batch microsatellite length distribution data in parallel is performed by scMSI to speed up operations.

![image](https://user-images.githubusercontent.com/81967713/217526985-232c0da1-5a05-47b6-866d-5a8d6be0073b.png)


Python package versions
--

scMSI works perfectly in the following versions of the python packages:

python3.7

scikit-learn 1.0.2

numpy  1.18.5

scipy 1.7.3

pandas 1.1.5

matplotlib  3.3.2

Example Usage
--

For example usage, please refer to the example folder, which contains test cases and test data. The core code is in the code folder
