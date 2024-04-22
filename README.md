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
- Insert descriptions here


Main Branch:
 - Final Metrics.ipynb notebook showcases the test results in depth between VGG16 Base Model and DenseNet201
 - Final_Project_MURA.ipynb notebook showcases the data preprocessed which includes analysis and data augmentation and best performaning model, DenseNet201 base with Transfer Learning technique
 - densenetmodel.keras incorporates the saved model version of densenet201 base from the Final_Project_MURA.ipynb notebook
 - test_w_preds.csv is a csv model of the testing results of the DenseNet base. It included image filepath, study type, patientID, correct label, and predicted label
 - test_w_preds_vgg.csv is a csv model of the testing results of the VGG16 base. It included image filepath, study type, patientID, correct label, and predicted label
 - vggModel.ipynb notebook encompassed the training, fitting, and testing of our second approach.
 - vggmodel.keras incorporates the saved model version of the vgg16 base from the vggModel.ipynb notebook


Vgg Branch:
Thia brancg was used to create the vgg16 base model. Files relate very closely to the main branch with the excpetion of it being slightly outdated due making our final adjustments of the VGG16on the main branch. It's also important to note the vgg16 version was running on the CS server which gave us technical errors at times and different results compared to the main brnach.
- vgg16_train_output.png shows visualization of the training results
- vgg16_test_w_preds.csv a csv file of the testing resuls with the vgg16 base


### Citations


----

#### Division of Labor

All group members contributed equally. Note that the GitHub commit history is slightly off and that code is all contained in main branch since we built off each others’ work.
- **Michelle**: Preprocessed and split training/validation/test data, conducted exploratory analysis, developed multiple visualizations, completed data augmentation
- **Amelia**: Defined DenseNet model, ran training/validation/test datasets, developed training/validation loss and accuracy graphs, calculated metrics
- **Jennifer**: Defined VGGNet model, ran training/validation datasets, developed training/validation loss and accuracy graphs, calculated metrics
- **Deva**: Created GitHub repo, explored optimizations, ran optimizations on both DenseNet and VGGNet models, and created GradCam visualizations
- **All**: Debugging on Discord, looking over each others’ work and offering improvements


