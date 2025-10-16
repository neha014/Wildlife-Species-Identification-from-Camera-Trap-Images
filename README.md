# Wildlife Species Detection using Camera Trap Images
To develop a system that automatically identifies and classifies wildlife species from camera trap images using image processing and machine learning techniques.

## Table of Contents
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Evaluation](#evaluation)
- [Contirbuting](#contributing)
- [Author](#author)

## Datasets
The dataset utilized in this project contains labeled images representing 10 animal species: Buffalo, Cheetah, Deer, Elephant, Fox, Jaguar, Lion, Panda, Tiger, and Zebra. The datasets can be accessed at:
- [Dataset 1](https://www.kaggle.com/datasets/biancaferreira/african-wildlife)
- [Dataset 2](https://www.kaggle.com/datasets/brsdincer/danger-of-extinction-animal-image-set)
- [Dataset 3](https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset )

## Project Structure
    
    Wildlife-Species-Detection/
    ├── .dvcignore
    ├── .gitignore
    ├── README.md
    ├── requirements.txt
    ├── yolov8n.pt
    │
    ├── .streamlit/
    │   └── config.toml
    │
    ├── config/
    │   └── custom.yaml
    │
    ├── images/
    │   ├── chee.jpg
    │   ├── ele.jpg
    │   ├── fox.jpg
    │   ├── lion.jpg
    │   ├── paan.jpg
    │   ├── rhino.jpg
    │   ├── tiger.jpg
    │   ├── turt.jpg
    │   └── ze.jpg
    │
    └── logs/
        └── log.log


## Getting Started
Follow theses steps to set up the environment and run the application.
1. Fork the repository [here](https://github.com/ldebele/animal-Species-Detection).
2. Clone the forked repository.
    ```bash
    git clone https://github.com/<YOUR-USERNAME>/Wildlife-Species-Detection
    cd Wildlife-Species-Detection
    ```

3. Create a python virtual environment.
    ``` bash
    python3 -m venv venv
    ```

4. Activate the virtual environment.

    - On Linux and macOS
    ``` bash
    source venv/bin/activate
    ```
    - On Windows
    ``` bash
    venv\Scripts\activate
    ```

5. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```
6. Run the application.
    ```python
    python -m streamlit run scripts/app.py

    ```

## Evaluation
The model’s performance is assessed using evaluation metrics including Precision, Recall, and Mean Average Precision.

| Model   | Precision | Recall | F1-score | mAP@0.5 | mAP@0.5:0.95 |
|---------|-----------|--------|----------|---------|--------------|
| YOLOv8  |   0.944   |  0.915 |   0.93   |   0.95  |    0.804     |


## Contributing
Open for contributions and suggestions, if you have any improvements, or bug fixes, feel free to open an issue or a pull request.

## Author
- `Neha M`
