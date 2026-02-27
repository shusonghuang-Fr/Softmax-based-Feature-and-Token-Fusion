Due to the test dataset being so large, we don't upload the dataset for testing. If you are interested, you can test your dataset with this code.
The input image size should be a numpy array with shape (samples, 224, 224, 3),  where 224 * 224 is the number of pixels, and 3 is the number of channels.
Firstly, we can run the low-level model Step1_feature_extraction.py,  Step1_token_extraction.py,  then in the high level, using the pretrained model for feature-token extraction, feature fusion, and further using train_step2_test.py

The code has been tested by our own multispectral images, the public dataset Plantvillage, opensource grapevine disease dataset from Kaggle, and the AI Challenge 2018.  
Relative results have been submitted to one journal. 

###Achknowledgement
Thanks to Guillaume Heller for sharing part of the code to have an idea for the low-level code.
In this code, to improve readability, the ChatGPT OpenAI tool is used to provide part of the code documentation. 
The design, idea, and its validation were implemented by the authors.
