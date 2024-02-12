## get started

(it is advised to install into a virtual environment)


> [!IMPORTANT]  
> library works with python version 3.8.x

to install the project library with all its dependencies:
```bash
pip install -e .
```


## project structure
```
├── newspapers/
|   ├── data/               #all of the data (training) for this project
|   |   ├── additional      #additional data saved during processing
|   |   ├── processed       #data or data objects saved during processing
|   |   ├── raw             #raw data (never edited)

|   ├── models/             #all saved models during trainign
|   |   ├── RUN_NAME        #directory created during a run, containing the saved model

|   ├── notebooks/          #all notebooks for analysis and visualization purposes

|   ├── results/            #all metrics for model analytics are saved here
|   |   ├── test/           #save metrics during tests
|   |   |   ├──RUN_NAME
|   |   ├── train           #save metrics during training
|   |   |   ├──RUN_NAME

|   ├── src/                #package root
|   |   ├── config          #where to store global configuration variables
|   |   ├── models          #where to define models
|   |   ├── preprocessing   #code for preprocessing data
|   |   ├── test            #code for testing
|   |   ├── train           #code for training various models
|   |   ├── utils           #utility functions
```
## example
> [!IMPORTANT]  
> all commands are run from the /src directory

### preprocessing the data

```bash
python preprocessing\main.py --data english_news.csv --name english_processed
```

4 files will be saved under `/data/processed/...` which will be `english_processed_targets.npy`, `english_processed_vocab.json`, Venglish_processed_tokens.npy`, `english_processed_bow.pt`

### train model

```bash
python train\W2Vmodel.py -ff english_processed_tokens.npy -tf english_processed_targets.npy -vf english_processed_vocab.json --run_name EXPERIMENT_1
```     

will train the W2V model with the previously preprocessed data and save the training results under `results\train\EXPERIMENT_1\...` and the test results under `results\test\EXPERIMENT_1\...` 


