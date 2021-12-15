# ML-IMTA-Project

In this repository you will find the final project for the UE Machine Learning (IMT Atlantique - 2021).

### Authors:

<ul>
  <li>Martina María BALBI ANTUNES</li>
  <li>Mateo BENTURA</li>
  <li>Ezequiel CENTOFANTI</li>
  <li>Kevin MICHALEWICZ</li>
</ul>

### Environement 

In order to be able to execute the following steps, you will need to create a Python 3 Environement.
This can be done by:

```bash
pip install requirements.txt
```

### Directory Structure

```
ML-IMTA-Project
│   README.md                                   # this file
│   requirements.txt
|   main.py                                     # main python file 
├── datasets
│   ├── data_banknote_authentication.txt        # banknote auth dataset
│   └── kidney_disease.csv                      # kidney disease dataset            
├── lib         
│   ├── clean_normalize.py                      # functions to clean and normalize datasets
│   ├── ml_functions.py                         # implementations of ML methods
│   ├── nn_util.py                              # some useful tools for Neural Networks
│   └── tools.py                                # some useful general tools
├── plots                                          
│   ├── confusion matrices
│   └── nn_loss                          
```

### Execution 

```bash
python main.py 
```