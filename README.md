A.	Contribution Table:
Name	Student ID	Contribution
Jeshfique Mahmud	2111097642	
- Designed the neural network architecture using ‘ResNeXt-50’  
- Implemented the ModifiedResNeXt class with custom classifier layers  
- Tuned hyperparameters: optimizer, learning rate, scheduler, and label smoothing  
- Led model training and validation over 25 epochs  
Syeda Nusaiba Sharifeen	2132576642
- Handled *dataset cleaning, splitting, and ‘custom DataLoader’ creation  
- Applied *advanced image augmentation techniques* for better generalization  
- Implemented ‘class balancing’ using WeightedRandomSampler  
- Evaluated model with ‘confusion matrix, ‘classification report, and ‘cross-validation’

Chadne.	2132112464
- Documented the methodology, training results, and augmentation strategies  
- Designed the *block diagram* representing the model's architecture  
- Created inference scripts for real-world testing of trained models  
- Compiled *result tables*, training logs, and final test accuracy summary  



B.	Data Augmentation method:
To enhance the performance and generalizability of the ResNeXt-50 architecture for classifying fake news images, we implemented the following data augmentation techniques in our model pipeline:
1.	Resizing and Cropping
•	Resize(256, 256) + RandomCrop(224) + RandomResizedCrop(224)
These steps ensure all images conform to the input size required by ResNeXt while introducing variability in scale and object positioning. This helps the model recognize patterns irrespective of size or location within the image.

2.	Geometric Transformations
•	RandomRotation(20)
Prepares the model to handle minor orientation changes, such as tilted headlines or faces.
•	RandomHorizontalFlip(p=0.5)
Teaches the model to be invariant to left-right flips, useful for mirrored but contextually similar images.
•	RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
Mimics real-world variations like camera movements, zooms, and distortions, enhancing the model’s adaptability.

 
 3. Color-Based Transformations
•	Color Jitter (brightness, contrast, saturation, hue) 
 Addresses variations in lighting, color tones, and image quality across fake/real news sources. This is critical since fake news images are often edited, filtered, or low quality.
•	RandomAdjustSharpness(1.5) 
 Makes the model robust against blurry or overly sharp images, common in manipulated content.
4.	Random Erasing
•	RandomErasing(p=0.3)
Simulates occlusions or tampering, forcing the model to rely on broader features rather than overfitting to specific regions. This is particularly relevant for fake news, where images may be partially obscured or doctored
5.	Normalization
•	ToTensor() + Normalize(mean, std)
Standardizes the data for numerical stability and aligns with ImageNet pretrained weights, facilitating faster convergence and better accuracy.

Why These Techniques Are Effective:

Our model addresses the complexities of distinguishing fake from real news images, which often exhibit:
•	Manipulations like cropping, filtering, or blurring.
•	Inconsistent lighting or resolution.
•	Non-standard formats and layouts.
These augmentations were selected to:
•	Increase dataset diversity.
•	Mitigate overfitting on limited training data.
•	Encourage the model to learn high-level visual features rather than memorizing specific details.
C.	Final Result:




Metric	  Fake	 Real	 Overall
Precision 0.81	0.80	0.80
Recall	  0.80	0.81	0.80
F1 Score	0.80	0.81	0.80
Accuracy			80.45%
Support 	1919	1918	3837
(Samples)





D.	Final Plan for Completing the Project
The project focuses on detecting multimodal news by text data and visual data (images) to improve accuracy and reliability. 
The image-based model is built using Resnext-50, while text data is processed by architecture based on a transformer (for example: Bert).
Two methods will be merged into a common model to classify whether it is real or wrong.




The final plan includes:
1.	Training the image-based model (ResNeXt-50) – already completed with 80.45% test accuracy.
2.	Fine-tuning the text-based model using transformer-based architecture.
3.	Integrating both models into a unified multimodal architecture for enhanced performance.
4.	Performing error analysis, visualization, and detailed evaluation (precision, recall, F1-score).
5.	Building an interface for real-time inference and testing with new samples.
6.	Finalizing documentation, diagrams, and report for submission.

