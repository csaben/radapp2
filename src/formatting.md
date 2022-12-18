File Tree

DCM/
├─ DCM/
│  ├─ Control/
│  │  ├─ P#/
│  │  │  ├─ S#/
│  │  │  │  ├─ S_type/
│  │  │  │  │  ├─ 1-01.dcm
│  │  │  │  │  ├─ 1-02.dcm
│  ├─ Tumor/
│  │  ├─ P#/
│  │  │  ├─ S#/
│  │  │  │  ├─ S_type/
│  │  │  │  │  ├─ 1-01.dcm
│  │  │  │  │  ├─ 1-02.dcm

Methodology

for each pathology (control vs Tumor) convert each S_type directory worth of dcm into an average image with the 
appropriate pathology label. Store as 2d npdarray np.arrray([Img],[0 or 1])

Modules

dataset.py
engine.py
train.py

Functions
ds_helper() ; given file directory, finds the S# dir, for:each s_type in each S# convert dcm to avg img, add to
              an array to return at end of function (effectively takes tumor or control and returns set of images)
dataset methods() ; init->use ds_helper , len->return number of images, get->take idx and return that image and label
		  ; in engine dataset should be declared twice, one for control and one for tumor dataloader
		  ; abhishek [example](https://github.com/abhishekkrthakur/bert-sentiment/blob/4283d293253ea4a3c868bcae0ae210b4033d2e1b/train.py)

---

based on abhi example;
to use dataloader need to properly return a img and label after being given all imgs and corresponding labels




