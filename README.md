# MURA-Classification
COMP 5300 Computing for Health & Medicine   
Final Project - Spring 2024  
Michelle, Deva, Amelia, Jennifer

### Musculoskeletal Abnormality Detection  
The purpose of this project is to use machine learning (deep learning) to develop an automated system to detect musculoskeletal abnormalities from radiographic X-Ray images.

Dataset: https://stanfordmlgroup.github.io/competitions/mura/

#### How to run code:
- Download dataset from following link and unzip folder
  - https://stanfordaimi.azurewebsites.net/datasets/3e00d84b-d86e-4fed-b2a4-bfe3effd661b
- Run Jupyter notebook code contained in Final_Project_MURA.ipynb for DenseNet201 base on the main branch
- Run Jupyter Notebook code contained in vggModel.ipynb for Vgg16 base

#### File descriptions

Main Branch:  

 - **Final Metrics.ipynb** notebook showcases the test results in depth between VGG16 Base Model and DenseNet201
 - **Final_Project_MURA.ipynb** notebook showcases the data preprocessed which includes analysis and data augmentation and best performaning model, DenseNet201 base with Transfer Learning technique
 - **densenetmodel.keras** incorporates the saved model version of densenet201 base from the Final_Project_MURA.ipynb notebook
 - **test_w_preds.csv** is a csv model of the testing results of the DenseNet base. It included image filepath, study type, patientID, correct label, and predicted label
 - **test_w_preds_vgg.csv** is a csv model of the testing results of the VGG16 base. It included image filepath, study type, patientID, correct label, and predicted label
 - **vggModel.ipynb** notebook encompassed the training, fitting, and testing of our second approach.
 - **vggmodel.keras** incorporates the saved model version of the vgg16 base from the vggModel.ipynb notebook


Vgg Branch:  

This branch was used to create the vgg16 base model. Files relate very closely to the main branch with the excpetion of it being slightly outdated due making our final adjustments of the VGG16on the main branch. It's also important to note the vgg16 version was running on the CS server which gave us technical errors at times and different results compared to the main brnach.
- **vgg16_train_output.png** shows visualization of the training results
- **vgg16_test_w_preds.csv** a csv file of the testing resuls with the vgg16 base


### Citations

Coding Links:
https://stackoverflow.com/questions/9727673/list-directory-tree-structure-in-python
https://www.kaggle.com/code/cafelatte1/alien-and-predator-classification-beginner/notebook
https://stackoverflow.com/questions/71572762/cannot-identify-image-file-io-bytesio-object-at-0x000001e6cf13a108
https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4
https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/
https://medium.com/geekculture/boost-your-image-classification-model-with-pretrained-vgg-16-ec185f763104
https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4
https://www.tensorflow.org/tutorials/keras/save_and_load
https://www.tensorflow.org/guide/keras/serialization_and_saving 
https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4
https://www.learndatasci.com/tutorials/hands-on-transfer-learning-keras/
https://medium.com/geekculture/boost-your-image-classification-model-with-pretrained-vgg-16-ec185f763104
https://towardsdatascience.com/transfer-learning-with-vgg16-and-keras-50ea161580b4

Almezhghwi, Khaled et al. “Convolutional neural networks for the classification of chest X-rays in the IoT era.” Multimedia tools and applications vol. 80,19 
(2021): 29051-29065. doi:10.1007/s11042-021-10907-y

Barondess, J., & Cullen, M. (2002). Musculoskeletal disorders and the workplace. International Journal of Epidemiology, 31(3), 702–702. 
https://doi.org/10.1093/ije/31.3.702

Cieza, Alarcos, et al. "Global estimates of the need for rehabilitation based on the Global Burden of Disease study 2019: a systematic analysis for the Global 
Burden of Disease Study 2019." The Lancet 396.10267 (2020): 2006-2017.

Keras. (n.d.). Keras documentation: Adam. https://keras.io/api/optimizers/adam/
Rajpurkar, Pranav, et al. "Mura: Large dataset for abnormality detection in musculoskeletal radiographs." arXiv preprint arXiv:1712.06957 (2017).

Sahota, H. (2023, November 30). An Intuitive Guide to convolutional neural networks. Comet.
https://www.comet.com/site/blog/an-intuitive-guide-to-convolutional-neural-networks

Shazia, A., Xuan, T.Z., Chuah, J.H. et al. A comparative study of multiple neural network for detection of COVID-19 on chest X-ray. EURASIP J. Adv. Signal 
Process. 2021, 50 (2021). https://doi.org/10.1186/s13634-021-00755-1
Worldometer. "World Population.” www.worldometers.info/world-population/




----

#### Division of Labor

All group members contributed equally. Note that the GitHub commit history is slightly off and that code is all contained in main branch since we built off each others’ work.
- **Michelle**: Preprocessed and split training/validation/test data, conducted exploratory analysis, developed multiple visualizations, completed data augmentation
- **Amelia**: Defined DenseNet model, ran training/validation/test datasets, developed training/validation loss and accuracy graphs, calculated metrics
- **Jennifer**: Defined VGGNet model, ran training/validation datasets, developed training/validation loss and accuracy graphs, calculated metrics
- **Deva**: Created GitHub repo, explored optimizations, ran optimizations on both DenseNet and VGGNet models, and created GradCam visualizations
- **All**: Debugging on Discord, looking over each others’ work and offering improvements


