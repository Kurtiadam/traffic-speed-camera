# Development of a speed measurement system using spatial transformer network and monocular depth estimation algorithm

This repository is for student work at the Budapest University of Technology and Economics' Scientific Student Conference.

![image](https://github.com/Kurtiadam/traffic-speed-camera/assets/98428367/b55f3062-624f-4441-bcc4-c568be6a9135)

## How to use 
Required libraries and other resources:
- download AdaBins pretrained models from here: https://drive.google.com/drive/folders/1nYyaQXOBjNdUJDsmJpcRpu6oE55aQoLA?usp=sharing, place it in a folder called "pretrained"
- create virtual environment with conda using the following command: ```conda env create -f environment.yaml```

Run the following commands:\
For inspecting the Brazilian speed measurement benchmark: ```python main.py --config_path .\config\config_brazil.yaml```\
For inspecting the license plate recognition: ```python main.py --config_path .\config\config_ocr.yaml```

Other commands can be viewed by: ```python main.py --help```

You can exit the running of the algorithm by pressing 'q' or waiting for the input to finish. Upon completion, the speed measurement results will be saved in an excel file named 'speed_estimation_results.xlsx'.
