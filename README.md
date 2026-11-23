<img width="1779" height="884" alt="image" src="https://github.com/user-attachments/assets/88681a3b-86bf-4665-9d5c-1343d5dcfe92" /># behavioural_cloninghttps://github.com/sergio-omar/behavioural_cloning/edit/main/README.md

![Uploading Pine-Lined Roadside Drive.png…]()


Definetly one of the most fun projects I've work on.
Personaly this project tought me how a small NN can copy the way I drive in a car simulator just by watching me how to do it. It was an eye opening on the capabilities of what a neural network can do. I know that there a projects using ML to tell the meaning of an image, but just watching this in a simulation, its like watching something to come alive in front of your eyes, and the fun part is that is created by you. Also one of the things that is surprising is that the process is not particularly complex rather intuitive. When I let the model to drive the simulator was an incredible experience, was like some sort of magic watching a tiny NN sorting the race. After this project I would like to take it further by trying new simulators like Mario kart or something alike.

Let's go to the process on how it was achieved.
1) Setup the environment. Install the unity simulator
2) Drive several times through the circuit, as perfect as possible (What I mean by perfect is for example not playing around spining the car, nor give very crazy turns, think as if you were teaching a human to drive.)
3) In my case I drive throuhg the circuit like 3 times.
4) Save all the data in CSV format. This files contain acceleration, stearing angle, throthle, break and 3 pictures (front, left, right) of the current frame in the unity simulator.
5) The python script that was on charge of learning the process, was achieved by using SKLearn framework. The steps to achieve this were: Extract the data from the CSV files. Split the data into train and validation. Build batches with the genarator. Accomodate some data for example to put the pixel values into 1 to 0 scale, and also cropping the image to avoid adding more noise. To be more specific I removed the sky. Also the CNN was build. The CNN used for this project was: 2 Convulotiona NN then it goes to an average pooling 2D then it is flatten to go through a 120 dense layer then a 84 dense layer with a 0.2 dropout to reduce overfitting. For the training I use MSE for the loss and adams for the optimizer. Also I ran it for 2 epochs.
6) After severla iteration of adjusting the model and driving manually I achieved a model that can drive through the scene without any help.


https://github.com/user-attachments/assets/de357173-3629-47de-8a23-4e593694ff77


