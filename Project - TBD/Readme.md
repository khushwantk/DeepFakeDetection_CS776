Group - TBD Truth Behind Deepfakes

Description of folders:
- Sample_data contains some FF++ videos to simulate our `01_dataset_split.py` and `02_face_extractor.py` code.
- model_weights contains our final trained model weights and are needed by  `inference.py`.
- results folder will contains saved data from running `run.py`.
- Image deepfake detection contains code for Phase 1 of our project.


Install the environment:
`pip install -r requirements.txt`

Alternatively

`pip install opencv-python torch torchvision facenet-pytorch numpy pandas scikit-learn matplotlib tqdm seaborn`

---
(Optional as we are providing link to dataset contaning extracted faces from FF++ dataset)


1. We download the FF++ dataset and create 2 folders real and fake contaning respective videos.
2. We do dataset split by running dataset_split.py by providing path to the folder cantaning real-fake folders. It will create a Splitted folder contaning the dataset split as train-val-test.
3. Pass the folder Splitted folder path to face_extractor.py to extract faces and save in Extracted folder.

This Extracted folder will be treated as our main dataset.

Structure of main dataset:
- train-val-test folders.
- Each of these will contain two folders real and fake.
- Each of real-fake folders will have many folders contaning respective 32 extracted faces.

**Link to our dataset**: <https://github.com/ondyari/FaceForensics>
Script to download FF++ dataset is also provided :faceforensics_download_v4.py

Usage:

`wget https://kaldir.vc.in.tum.de/faceforensics_download_v4.py`

`python3 faceforensics_download_v4.py /content/FaceForensics -d all -c c23 -t videos --server EU2`

We have extracted the faces and link to the dataset :
<https://www.kaggle.com/datasets/khushwantk/newdataset>

---

**Training and Testing** :
1. Execute run.py

    `python3 run.py --model swin`

    Other parameters are set to some default values. If you want to change you can pass --epochs, --batch_size, --patience, --num_frames flags.

    num_frames has to be <32.
    model choices=["efb0", "resnext", "swin"]

    Example usage :

    `python3 run.py --model swin --epochs 2 --batch_size 2 --patience 5 --num_frames 10`

2. To execute the UI

    `python3 inference.py`

3. To check generalization of models on other datasets ie UADFV and CelebDFv1

    `python3 check_gen.py --model <model name> --model_weights <path to weights> --test_data <path to test data>`

    Give same model_name and model_weights ie efb0-efb0_weights otherwise you will get errors.
