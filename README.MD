produce-shelf-life-estimator/
│
├── data/
│   ├── raw/                      # Your original folders (unripe, ripe, etc.)
│   └── processed/                # Where your generated CSV is saved
│
├── src/                          # Your core modular python scripts
│   ├── __init__.py
│   ├── build_dataset.py          # The script we just wrote to generate the CSV
│   ├── model.py                  # The MobileNetV2 architecture function
│   ├── train.py                  # The script that loads data and runs model.fit()
│   └── export.py                 # Script to convert the trained model to TFLite
│
├── models/                       # Where your trained networks are saved
│   ├── checkpoints/              # Saves model weights during training
│   └── shelf_life_edge.tflite    # The final quantized model for the app
│
├── app/                          # Your frontend deployment (React Native/Flutter/Android)
│   └── ...                       # App source code goes here
│
├── requirements.txt              # List of dependencies (tensorflow, pandas, opencv-python)
└── README.md                     # Your project documentation