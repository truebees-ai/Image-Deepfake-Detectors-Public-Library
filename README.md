# TITOLO

# DeepShield Dataset
The dataset is available at this [link](https://zenodo.org/records/15648378)


# Detectors

## Use pretrained network
The R50-E2P network was trained according to the information provided in the paper on a limited subset of non-shared data.\
If you plan on using the network for other purposes other than to replicate the results in the paper, you should re-train it on a more comprehensive subset.

Create directory with ```mkdir -p ./detector/train/gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre/models/```.\
And put the [pretrained weigths](https://drive.usercontent.google.com/download?id=1m0b8HZuOCcF_-l2-GJBi3C22RuG5UDvU&export=download) inside, make sure that the file is named ```best.pt```.

## Set up Virtual-Env
```
conda env create -f environment.yml
```

## Run the code
Run the code inside ```./detector/```\
We suggest to use the provided launcher to automate training and test phases:
```
python launcher.py
```

Alternatively you can make individual calls using ```train.py``` and ```test.py```:
```
python train.py --name "gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre" --split_file ./split.json --task train --num_threads 8 --data_keys "gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre" --data_root DATASET_ROOT --device cuda:0 --arch nodown --prototype --freeze
python test.py --name "gan2:pre&sdXL:pre&realFFHQ:pre&realFORLAB:pre" --split_file ./split.json --task test --num_threads 8 --data_keys DATA_ID --data_root DATASET_ROOT --device cuda:0 --arch nodown --prototype --freeze
```
If you call ```train.py``` and ```test.py``` directly, you have to specify ```DATASET_ROOT``` in both and ```DATA_ID``` for the test code, according to [this guideline](detector/README.md).

# Results


