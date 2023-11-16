# Speed camera algorithm with computer vision

This repository is for student work at the Budapest University of Technology and Economics' Scientific Student Conference.

![image](https://github.com/Kurtiadam/speedcam_cvs/assets/98428367/51f2c69d-5758-4fde-9175-7b40dd9e4655)

## How to use 
Required libraries and other resources:
- download AdaBins pretrained models from here: https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing, place it in a folder called "pretrained"
- create virtual environment with conda using the following command: conda env create -f environment.yaml

Run the following commands:
For inspecting the speed measurement benchmark: python main.py --input_path ./samples/speed_measurement_sample_cut.MOV
For inspecting the license plate recognition: python main.py --input_path ./samples/ocr_sample.MOV

Other commands can be viewed by: python main.py --help

You can exit the running of the algorithm by pressing 'q' or waiting for the input to finish. Upon completion, the speed measurement results will be saved in an excel file named 'speed_estimation_results.xlsx'.
