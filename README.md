Due to the large size of the test dataset, we do not offer an upload service for it. If you are interested, you can use this code to test your own dataset.

###Simple implementation: 

1. Prepare the image size. The input image size should be a NumPy array of shape (samples, 224, 224, 3), where 224 * 224 is the number of pixels, and 3 is the number of channels.
2. We can run the low-level model 'Step1_feature_extraction.py' for feature extraction, and run 'Step1_token.py' for token extraction.
3. At higher levels, pretrained models are used for token extraction, feature-token fusion, and then train_step2_test.py is used to test.

The code has been tested by our own multispectral images, the public dataset Plantvillage, opensource grapevine disease dataset from Kaggle, and the AI Challenge 2018.  
Relative results have been submitted to one journal. 

###Achknowledgement

Thanks to Guillaume Heller for sharing part of the code to have an idea for the low-level code. The project is supported by the Grand-Est Region, France.
In this code, we use ChatGPT openAI to make the code more concise and easier to understand. 
The design, idea, and its validation were implemented by the authors.
