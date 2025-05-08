Script file : image_final.ipynb
Model weights : final_efficientnet_b0.pth

To install requirements : pip install pandas torch torchvision tqdm matplotlib gradio scikit-learn seaborn pillow

Steps :

1. Download the dataset and store in the same directory and the directory structure of the dataset should be like "kagglehub/datasets/xhlulu/140k-real-and-fake-faces/versions/2/real_vs_fake/real-vs-fake"

    Dataset link : https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

2. Use "make_dataset_csv.ipynb" to generate csv file of train, valid and test dataset. On running all cell it should generate 3 csv files with name :
    train_labels.csv
    valid_labels.csv
    test_labels.csv

3. Run all the cells of "image_final.ipynb" and all the training and testing will be done.

Note : Since model weights are given for prediction no need to train again just run all cell of image_final.ipynb except the one which has __main__ func. Then run the Run web interface cell.
Although gradio based web interface will shown in the notebook itself, but you can click on the localhost link provided to test on the default browser.

