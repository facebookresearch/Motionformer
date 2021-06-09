#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import csv
import glob
import os


def k400_preproces(
    root_dir='/datasets01/kinetics/070618/400/', split_dir='train_avi-288p', mode='train'
):
    data_prefix = os.path.join(root_dir, split_dir)
    files = list(sorted(glob.glob(os.path.join(data_prefix, '*', '*'))))
    classes = list(sorted(glob.glob(os.path.join(data_prefix, '*'))))
    classes = [os.path.basename(i) for i in classes]
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    with open(f'{mode}.csv', mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for path in files:
            class_name = path.split('/')[-2]
            class_idx = class_to_idx[class_name]
            csv_writer.writerow([path, class_idx])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='K-400 preprocessing')

    parser.add_argument(
        '--root_dir', 
        default='/datasets01/kinetics/070618/400/', 
        type=str, 
        help='root dir of K-400 folder'
    )
    parser.add_argument(
        '--split_dir', 
        default='train_avi-288p', 
        type=str,
        help='name of dir of split'
    )
    parser.add_argument(
        '--mode', 
        default='train', 
        type=str,
        help='name of dir of split'
    )
    args = parser.parse_args()
    k400_preproces(
        root_dir=args.root_dir, 
        split_dir=args.split_dir,
        mode=args.mode
    )