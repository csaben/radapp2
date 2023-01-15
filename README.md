input directory is not uploaded due to bloat.

# For Yonathan

1. unzip the zip file in the following directory
```
prev_input/img_dataset/
```

2. remove zip file and "dataset" folder. you only need 
```
prev_input/img_dataset/yes
prev_input/img_dataset/no
```

3. To generate a model and the augmented images dataset run 
```
prev/training.py
```

4. set your MODEL variable in config.py to match the one you want OR just unzip the zipped model file I left for you in the prev_input/model and the script will use that

5. To generate a prediction based on a given patient folder (e.g. S7) run 
```
prev/glioma.py
``` 
AFTER updating the input_path to point at the patient you want (i will make a way to do this from the terminal as a flag later on)

* note that to make new models you should go into the training.py and adjust the name (i can automate this later if needed)
* note the preprocessing was super intensive so I have made it the bare minimum for now as a proof of concept (model still trains to 70% accuracy or so)
* the directory storing patients is in prev_input/test_workdir/data/in (unzip)

As a final note for now, I have left only a single patient (S7)
that is hardcoded to be used on inference. I will update that code
and also include more patient data later but this leaves you with proof of concept;
- training script
- inference script
- .json bucket file
- img_dataset yes and no buckets
- sample inference for dcm folder bucket

this first addition is very much proof of concept and good reference work,
in the future I'll be starting from scratch for other scan types and datasets.
this current POC outlines the need for delineating a jpg dataset for trainig initially
and a way to convert new dcm files into that pile and retraining







