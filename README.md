# RG3

_**Thank you for reviewing our paper!**_

Here we provide the code of RG3. Our work is submiited to SIGKDD 2024. 

We are willing to update the code and instructions if there are any concerns!

## Code Instruction:

First enter the folder of src:

```
cd ./src
```

We first need to train the encoder:

```
python main_encdoer.py dataset=sbm +experiment=sbm.yaml
```

Then we train the graph generation model:

```
python main.py dataset=sbm +experiment=sbm.yaml
```


To choose specific parameters, modify the dataset and experimental config in the input command
