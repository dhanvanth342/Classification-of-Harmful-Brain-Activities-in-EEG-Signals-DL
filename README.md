

# Classification of Harmful Brain Activities in EEG Signals with Deep Learning

## 📑 Summary

This project aims to automate the classification of harmful brain activity patterns in EEG signals to assist healthcare professionals in identifying seizure-related and other brain-damaging activities more efficiently. Utilizing deep learning models, our analysis helps mitigate the time, cost, and error-prone nature of manual EEG reviews, providing a tool for quicker and more reliable diagnoses.

**Use Case**: Rapid, automated detection of harmful brain patterns in EEG data can enable faster intervention in critical care settings, reducing neurologists' workload and improving patient outcomes.

**Approach**: 
We trained and tested three state-of-the-art models—**ResNet50**, **EfficientNet B0**, and **InceptionNet V3**—on a preprocessed dataset of annotated EEG signals. Key data preprocessing steps included file mapping, normalization, log transformation, and mean subtraction to enhance feature clarity and reduce computational load. Hyperparameters were optimized to balance performance and computational efficiency.

**Conclusion**: 
Among the models, **ResNet50** provided the highest accuracy (61.004%), while **InceptionNet V3** demonstrated balanced performance across recall and precision metrics. The results underscore the challenges of working with unbalanced datasets in medical imaging and suggest future improvements through data augmentation techniques to address this imbalance effectively.

---

## 🗂️ Code Sections and Implementation Details

### 1. **Data Preprocessing**

   - **File Mapping**: Organized the EEG and spectrogram data files using a unique identifier mapping, facilitating easy retrieval for model input.
   - **Normalization**: Scaled data to a range of (0, 1) to reduce computational intensity and standardize input features.
   - **Log Transformation**: Applied to enhance feature visualization by balancing the data range, reducing the effect of outliers.
   - **Mean Subtraction**: Centered data around zero to improve stability, preventing model bias.

### 2. **Model Deployment**

   - **Models Used**:
     - **ResNet50**: Incorporates skip connections to counteract vanishing gradient issues, enabling the model to learn both local and global features efficiently.
     - **EfficientNet B0**: Employs compound scaling, adjusting depth, width, and resolution for computational efficiency while retaining performance.
     - **InceptionNet V3**: Uses inception modules to capture multi-scale features with parallel convolutions, balancing depth and width.
   - **Training Environment**: All models were fine-tuned using **ImageDataGenerator** in TensorFlow with Keras as the backend on Kaggle GPU.

### 3. **Performance Metrics**

   - **Accuracy**: Measures the ratio of correct predictions to total predictions, best achieved by ResNet50 at 61.004%.
   - **Precision**: Evaluates the ratio of true positive predictions within all positive predictions; higher for InceptionNet V3.
   - **Recall**: Assesses true positive predictions out of all actual positive instances; InceptionNet V3 again scored highest.
   - **Balanced Accuracy**: Ensures unbiased class representation in the model's evaluation, crucial given the unbalanced dataset.

### 4. **Hyperparameters & Training Optimization**

   - **Hyperparameters**:
     - Loss Function: **Categorical Cross-Entropy**
     - Optimizer: **Adam**
     - Dropout: **30%** for regularization
     - Batch Size: **32** to balance computation and overfitting
     - Learning Rate: Dynamic range of **0.0001 – 0.00001**
   - **Optimization Techniques**:
     - **ReduceLRonPlateau**: Lowers the learning rate when validation accuracy plateaus.
     - **Early Stopping**: Prevents overfitting by stopping training when validation accuracy no longer improves.
     - **Dropout Layers**: Randomly drops 30% of parameters during training to improve generalization.

### 5. **Results & Observations**

   - **Training Performance**: ResNet50 displayed early signs of overfitting, whereas InceptionNet V3 improved consistently across epochs.
   - **Evaluation Findings**: While ResNet50 yielded the highest accuracy, InceptionNet V3 performed better in metrics like precision, recall, and balanced accuracy, highlighting the need for balancing in data-intensive applications.

### 6. **Conclusion**

This analysis provided insights into the advantages and limitations of popular deep learning architectures for medical data classification. **Future work** could include applying augmentation strategies to address dataset imbalance, potentially boosting model robustness for deployment in clinical environments.



