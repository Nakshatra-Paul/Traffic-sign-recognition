# Traffic-sign-recognition
Trained on the CIFAR10 dataset, model is used to recognize traffic signs frooomm user uploaded library
Traffic Sign Recognition

Overview
Traffic Sign Recognition is a machine learning project that aims to recognize and classify traffic signs from images. The project utilizes Convolutional Neural Networks (CNNs) for image classification and Object Detection models for locating traffic signs within images.

Features
Traffic Sign Classification: The project uses a trained CNN model to classify traffic signs into predefined categories.
Object Detection: Object Detection techniques are employed to locate and visualize traffic signs within images.
Pre-Trained Models: Pre-trained CNN models and Object Detection models are utilized for efficient and accurate recognition of traffic signs.
User Interface: A user-friendly interface allows users to upload images and visualize the results of traffic sign recognition.
Setup
Clone Repository: Clone the repository to your local machine using the following command:

bash

git clone https://github.com/your_username/traffic-sign-recognition.git
Install Dependencies: Install the required dependencies by running the following command:


pip install -r requirements.txt
Download Pre-Trained Models: Download the pre-trained CNN models and Object Detection models and place them in the appropriate directories within the project.

Usage
Traffic Sign Classification:

Run the classify_traffic_sign.py script to classify traffic signs from images.
Provide the path to the input image as a command-line argument.
Object Detection:

Run the detect_traffic_sign.py script to detect and locate traffic signs within images.
Provide the path to the input image as a command-line argument.
User Interface:

Launch the user interface by running the app.py script.
Upload images using the provided interface and visualize the results of traffic sign recognition.
Contributing
Contributions are welcome! If you would like to contribute to the project, please follow these steps:

Fork the repository.
Create a new branch (git checkout -b feature/new-feature).
Make your changes and commit them (git commit -am 'Add new feature').
Push to the branch (git push origin feature/new-feature).
Create a new Pull Request.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements
This project was inspired by the need for accurate traffic sign recognition in autonomous driving systems.
Special thanks to the creators of open-source CNN and Object Detection models used in this project.
