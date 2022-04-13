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
## Running Terminal
```
python run_terminal --input <INPUT_IMG_DIR> --trimap <TRIMAP_IMG_DIR> --gt <GT_IMG_DIR> 
                    --window-size 40 --sigmac 0.01 --sigmag 10 --minN 40 --max-iter 200
                    --min-like 1e-6 --cpu-count 8 --output-name OUTPUT_FILE.PNG
```

