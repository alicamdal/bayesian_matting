# bayesian_matting
Bayesian Matting Python implementation for Computational Methods

# Installing Libararies
All required libraries can be found in requirements.txt
## Using Pip
```
pip install -r requirements.txt
```
## Using Conda
```
conda install --file requirements.txt
```

# Running
There are two options for running application. 
## Running GUI
```
pyhton run_gui.py
```
![alt text](scripts/Samples/gui.png "GUI")

As can be seen above, GUI has textboxes for Bayesian parameters. These parameters can be changed by them. If application is desired to run without multiprocessing, cpu count should be set as 1.
## Running Terminal
```
python run_terminal --input <INPUT_IMG_DIR> --trimap <TRIMAP_IMG_DIR> --gt <GT_IMG_DIR> 
                    --window-size 40 --sigmac 0.01 --sigmag 10 --minN 40 --max-iter 200
                    --min-like 1e-6 --cpu-count 8 --output-name OUTPUT_FILE.PNG
```
Terminal application can be run as above. Reqired arguments are input, trimap and groundtruth images. Other arguments have their own default value.
If application is desired to run without multiprocessing, cpu count argument should be set as 1. Parser method can be seen below for default parameters.
![alt text](scripts/Samples/parser.png "Parser")
