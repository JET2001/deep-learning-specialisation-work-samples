# deep-learning-specialisation-work-samples

Here is a collation of some exercises from the DeepLearning.AI's [Deep Learning](https://www.deeplearning.ai/) Specialization.

This Specialization is hosted on [Coursera](https://www.coursra.org), and it consists of 5 courses:

1. Neural Networks and Deep Learning
2. Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization
3. Structuring Machine Learning Projects
4. Convolutional Neural Networks
5. Sequence Models

### Work Samples

Here are some assignments that serve as **work samples** of my course work.

1. **Regularization using Numpy**

   - This assignment involves making improvements to a neural network in Numpy, implementing the cost function and backpropagation with L2 Regularization, and forward prop with L2 Regularization.

2. **Optimization using Numpy**

   - This assignment involves making improvements to gradient descent in numpy applying stochastic gradient descent, momentum, RMSProp and Adam, and use random minibatches with moment optimizers, fixed and scheduled learning rate decay, to accelerate convergence and improve optimizaion

3. **Object Detection Principles using YOLO Algorithm in TensorFlow**

   - Using a sample dataset by Drive.AI, this assignment involves using images taken using a car's front-view camera to output bounding boxes of object classes on the road. The algorithm detects 80 types of object classes, and outputs a bounding box for each detected object.
   - This assignment uses the output of the YOLO Algorithm, performs Intersection over Union, and Non-Max Suppression, and filters the input boxes.

4. **Image Segmentation using U-Net in TensorFlow**

   - This assignment involves building our own U-Net architecture and implement semantic image segmentation on CARLA self-driving car dataset. We apply sparse categorical crossentropy for pixelwise prediction.

5. **Facial Recognition and Verification in TensorFlow**

   - This assignment involves implementing **one-shot learning** to solve a face recognition problem, and apply the triplet loss function to learn a network's parameters in the context of face recognition.
   - We use a pretrained Siamese Network (FaceNet) to compute 128-dimensional embeddings, and perform face verification and recognition with these encodings.

6. **Jazz Music Generation using LSTMs in TensorFlow**

   - In this assignment, we apply a one-to-many LSTM to a music generation task to generate your own Jazz Music with Deep Learning. This music generation system corresponds to 90 unique values. Each music note is mapped to some numerical index.
   - The LSTM model would take in a zero-vector as input, and the outputs $y^(<i>)$ is then passed as input to the block $(i+1)$-th LSTM along with the activation $a^{<i>}$.
   - The length of music for this exercise is fixed at 30 values.

7. **Debiasing Word Embedding Vectors**

   - Word Embeddings are computationally expensive to train. Because of underlying biases in the data, similar biases can be reflected in a word embeddings, and we can debias word vectors by means of some algorithms.
   - **Neutralization** - For non-gender specific words, debiasing word vectors involves treating the bias direction as a subspace, and to zero out the component of the embedding in the direction of the subspace, while keeping its orthogonal complement constant.
   - **Equalization** - For gender specific words, we can do an equalization algorithm for those. Equalization is applied to pairs of words that you might want to have differ only through the gender properly. Suppose that "actress" is closer to "babysit" than "actor". By applying neutralization to "babysit", you can reduce the gender stereotype associated with baby sitting, but still does not guarantee that "actor" and "actress" are equidistant from "babysit".
   - This assignment involves implementing the **neutralization** and **equalization** steps.

8. Emojifying Text using GloVE Embeddings and LSTMs in TensorFlow

   - This assignment creates a baseline model using word embeddings, and improves it by incorporating an LSTM.
   - The improved model involves a 2-layer LSTM network that passes the embedding of each word in the sequence into an LSTM as input, and applying a Dropout layer between the Layer 1 LSTM and Layer 2 LSTM. It outputs $y\hat$, that represents the emoji for that particular sentence.

9. Neural Machine Translation using Attention Model in TensorFlow

   - This assignment involves creating a model to translate human readable dates (20th January 2023) into machine readable dates (2023-01-20). The network will learn to output dates in the common machine readable format of "YYYY-MM-DD".
   - The human readable date is first passed into a Pre-Attention Bidirectional LSTM, which passes on the computed hidden-states into the attention model, which computes the $context$ vector for the timesteps. The $context$ vector for each timestep is passed as input into the Post-Attention Uni-Directional LSTM that has 10 outputs.

10. Transformer Architecture in TensorFlow
    - This assignment involves building and training a transformer model by creating positional encodings to capture sequential relationships in data, calculate scaled dot-product self-attention using word embeddings.
    - We also implement multi-head attention mechanism in the encoder and decoder (masked multi-head attention)

### More about the Specialisation

1. **Neural Networks and Deep Learning** - This course involved implementing feedforward neural networks in Numpy, Gradient Descent, and introduces the Neural Network Architecture.
2. **Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization** - This course covered strategies to improve neural networks via hyperparameter tuning:
   - reducing overfitting in neural networks via regularization techniques (Dropout/L2/BatchNorm)
   - to speed up convergence from Gradient Descent using optimization techniques (Minibatch/Exponentially Weighted Averages/RMS Prop/Adam/Learning Rate Decay)
3. **Structuring Machine Learning Projects** - this course introduces ML strategies such as orthogonalization, having optimizing and satisficing metrics, carryout error analysis and handling mixed dataset distributions. This course also analyses the conditions for multi-task learning, transfer learning and end-to-end deep learning.
4. **Convolutional Neural Networks** - This course covered CNNs through analysing Computer Vision applications:
   - through analysis of seminal CNN architectures (Inception Networks, AlexNet, MobileNet)
   - object detection and localization techniques (non-max suppression, intersection over union, anchor boxes, YOLO Algorithm)
   - face recognition and verification techniques (one-shot learning, Siamese Networks, Triplet Loss)
   - neural style transfer (Style and Content cost functions)
5. **Sequence Models** - This course covered Sequence Models through analysing fundamentals (RNNs, GRUs, LSTMs, Attention RNN and Transformers) as well as several applications (Music Generation, Emojifying plaintext, Trigger Word Detection and Machine Translation).
