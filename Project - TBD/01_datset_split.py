# Input must be a folder having real-fake folders. It will split it to train test val

import os
import shutil
from sklearn.model_selection import train_test_split

def create_split(src_root, dest_root, exts=None):
    """
    Create 70-20-10 split (train-val-test) for datasets organized as video files
    in 'real/' and 'fake/' folders.

    Args:
        src_root (str): path containing 'real/' and 'fake/' subdirs of video files
        dest_root (str): path where 'train/', 'val/', 'test/' splits will be created
        exts (tuple): optional tuple of allowed video extensions, e.g. ('.mp4', '.avi')
    """
    os.makedirs(dest_root, exist_ok=True)

    for label in ['real', 'fake']:
        src_label_dir = os.path.join(src_root, label)
        # collect files (optionally filter by extension)
        all_files = [
            f for f in os.listdir(src_label_dir)
            if os.path.isfile(os.path.join(src_label_dir, f))
            and (exts is None or f.lower().endswith(exts))
        ]

        # 1) 70% train, 30% temp
        train_files, temp_files = train_test_split(
            all_files, test_size=0.3, random_state=42
        )

        # 2) from temp, 2/3 → val (≈20% of total), 1/3 → test (≈10%)
        val_files, test_files = train_test_split(
            temp_files, test_size=1/3, random_state=42
        )

        # helper to copy a list of files into split/label
        def copy_split(split_name, file_list):
            dest_dir = os.path.join(dest_root, split_name, label)
            os.makedirs(dest_dir, exist_ok=True)
            for fname in file_list:
                src_path = os.path.join(src_label_dir, fname)
                dst_path = os.path.join(dest_dir, fname)
                shutil.copy2(src_path, dst_path)

        copy_split('train', train_files)
        copy_split('val',   val_files)
        copy_split('test',  test_files)

if __name__ == "__main__":
    # if you only want .mp4 and .avi videos, pass exts=('.mp4', '.avi')
    create_split(
        src_root='Sample_data/FF++_videos',
        dest_root='Sample_data/FF++_videos/Splitted',
        exts=('.mp4', '.avi', '.mov')
    )
