# DL
Deep Learning Assignment 2

# Q1

Prepare one python notebook (recommended- use Google Colab) to build, train and evaluate model (TensorFlow or TensorFlow.Keras library recommended) on the two datasets given below. Read the instructions carefully. 

__Question:__ Image Captioning: Image Captioning is the process of generating a textual description of an image. It uses both Natural Language Processing and Computer Vision to generate the captions. The dataset will be in the form [image → captions]. The dataset consists of input images and their corresponding output captions.


#### Encoder

The Convolutional Neural Network(CNN) can be thought of as an encoder. The input image is given to CNN to extract the features. The last hidden state of the CNN is connected to the Decoder.

#### Decoder

The Decoder is a Recurrent Neural Network(RNN) which does language modeling up to the word level. The first time step receives the encoded output from the encoder and also the <START> vector.


(10 marks) 

- Import Libraries/Dataset (0 mark) 
- Import the required libraries
- Check the GPU available (recommended- use free GPU provided by Google Colab). 
 

## Data Processing(1  mark) 
 
Read the pickle file (https://drive.google.com/file/d/1A8g74ohdb_5d2fPjc72yF7GxufE9GRcu/view?usp=sharing) (Links to an external site.) and convert the data into the correct format which could be used for the ML model. The pickle file contains the image id and the text associated with the image.

Eg: '319847657_2c40e14113.jpg#0\tA girl in a purple shirt holds a pillow.

Each image can have multiple captions.

319847657_2c40e14113.jpg -> image name

#0 -> Caption ID

\t  -> separator between Image name and Image Caption

A girl in a purple shirt hold a pillow . -> Image Caption

Corresponding image wrt image name can be found in the image dataset folder.

__Image dataset Folder :__ https://drive.google.com/file/d/1-mPKMpphaKqtT26ZzbR5hCHGedkNyAf1/view?usp=sharing (Links to an external site.) 

Plot at least two samples and their captions (use matplotlib/seaborn/any other library). 
Bring the train and test data in the required format. 
 

## Model Building (4 mark) 
Use Pretrained Resnet-50 model trained on ImageNet dataset (available publicly on google) for image feature extraction.
Create 4 layered LSTM layer model and other relevant layers for image caption generation.
Add L2 regularization to all the LSTM layers. 
Add one layer of dropout at the appropriate position and give reasons. 
Choose the appropriate activation function for all the layers. 
Print the model summary. 
 

## Model Compilation (0.5  mark) 
Compile the model with the appropriate loss function. 
Use an appropriate optimizer. Give reasons for the choice of learning rate and its value. 
Model Training (1 mark) 
Train the model for an appropriate number of epochs. Print the train and validation loss for each epoch. Use the appropriate batch size. 
Plot the loss and accuracy history graphs for both train and validation set. Print the total time taken for training. 
Model Evaluation (1 mark) 
Take a random image from google and generate caption for that image.




# Q2) 2.5 Marks

Let us define a sequence parity function as a function that takes in a sequence of binary inputs and returns a sequence indicating the number of 1’s in the input so far; specifically, if at time t the 1’s in the input so far is odd it returns 1, and 0 if it is even. For example, given input sequence [0, 1, 0, 1, 1, 0], the parity sequence is [0, 1, 1, 0, 1, 1]. 
 

Implement the minimal vanilla recurrent neural network to learn the parity function. Explain your rationale using a state transition diagram and parameters of the network.


## Evaluation Process -

- Task Response and Task Completion- All the models should be logically sound and have decent accuracy (models with random guessing, frozen and incorrect accuracy, exploding gradients, etc. will lead to deduction of marks. Please do a sanity check of your model and results before submission).
- There are a lot of subparts, so answer each completely and correctly, as no partial marks will be awarded for partially correct subparts.
- Implementation- The model layers, parameters, hyperparameters, evaluation metrics, etc. should be properly implemented.
- Notebooks without output will not be considered for evaluation.
-Request to upload an HTML version of the jupyter notebook as well with all the outputs and comments.

### Additional Tips -

- Code organization- Please organize your code with correct line spacing and indentation, and add comments to make your code more readable.
- Try to give explanations or cite references wherever required.
- Use other combinations of hyperparameters to improve model accuracy.
