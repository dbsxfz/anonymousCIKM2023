## README
Code for the article `Improve Deep Forest with Learnable Layerwise Augmentation Policy Schedules` submitted to CIKM'23
### Introduction
This README provides an overview of the code used in our research paper. It explains the purpose of each file and describes the libraries and versions we utilized. The code implements two models, `deep_forest.py` and `gcForestcs.py`, which are modified versions capable of utilizing augmentation. The `layer.py` file contains the code for constructing each layer of the model, including the implementation of random erase functionality. The `aug.py` file contains our implemented search augmentation policy and its application in a complete model. The `one_df` class can be used to train a DF (Deep Forest) or its variants with the provided policy or without any policy. The `aug_df` class performs policy search and records the selected policies.

### Libraries Used
The following libraries were used in our code:

| Library       | Version |
|---------------|---------|
| Python        | 3.9.16  |
| scikit-learn  | 1.2.0   |
| deep-forest   | 0.1.7   |
| numpy         | 1.22.4  |

### File Descriptions

1. `deep_forest.py` - This file contains the modified version of the Deep Forest model that supports augmentation. 
2. `gcForestcs.py` - This file contains the modified version of the gcForest model that supports augmentation.
3. `layer.py` - This file includes the code for constructing each layer of the model and implements the random erase functionality.
4. `aug.py` - This file contains our implemented search augmentation policy and its integration into a complete model. It includes the `one_df` class, which can train a DF or its variants with the provided policy or without any policy, and the `aug_df` class, which performs policy search and records the selected policies.
5. `train_augDF.py` - This file outlines the complete workflow for data preprocessing, model selection, and augmentation policy usage. To use the code with other datasets, you need to add dataset loading and preprocessing steps in `train_augDF.py`. Similarly, if you want to utilize other discovered augmentation strategies, modifications should be made in `train_augDF.py`.
6. `run.sh` - This shell script provides the complete training code for the `arrhythmia` dataset as an example. It trains the model five times, computes the mean and variance of the results, and redirects the output to the `records` folder for further examination.

### Additional Notes
- To make the code compatible with other databases, you need to add dataset loading and preprocessing steps in `train_augDF.py`.
- If you want to utilize other discovered augmentation strategies, modifications should be made in `train_augDF.py`.
- After discovering augmentation strategies, we found that abandoning the additional five-fold cross-validation can make better use of the data and achieve improved results. We have included the implementation code in `deep_forest_schedule.py`.

Please feel free to reach out if you have any further questions or need additional information.

### Testing Note
In our experiments, we performed testing and result recording after training each layer to further save time. We assure you that there is no leakage of the test set during this process. However, if you have any concerns, you can refer to the `test_arrhythmia.ipynb` notebook provided, which includes code for retesting after training. Simply comment out all code related to the test set during the model training process.

Please feel free to reach out if you have any further questions or need additional information.