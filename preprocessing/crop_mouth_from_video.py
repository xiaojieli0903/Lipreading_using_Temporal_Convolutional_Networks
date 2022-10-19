import argparse
import glob
import os

from dataloader import AVSRDataLoader
from utils import save2npz


def load_args(default_config=None):
    parser = argparse.ArgumentParser(description='Lipreading Pre-processing')
    # -- utils
    parser.add_argument('--video-direc',
                        default=None,
                        help='raw video directory')
    parser.add_argument('--landmark-direc',
                        default=None,
                        help='landmark directory')
    parser.add_argument('--filename-path',
                        default='./lrw500_detected_face.csv',
                        help='list of detected video and its subject ID')
    parser.add_argument('--save-direc',
                        default=None,
                        help='the directory of saving mouth ROIs')
    # -- convert to gray scale
    parser.add_argument('--convert-gray',
                        default=False,
                        action='store_true',
                        help='convert2grayscale')
    # -- test set only
    parser.add_argument('--testset-only',
                        default=False,
                        action='store_true',
                        help='process testing set only')
    # -- id
    parser.add_argument('--id', default=0, type=int, help='id')

    args = parser.parse_args()
    return args


args = load_args()

dataloader = AVSRDataLoader(convert_gray=args.convert_gray)

modality = "video"

lines = open(args.filename_path).read().splitlines()
lines = list(filter(lambda x: 'test' == x.split('/')[-2],
                    lines)) if args.testset_only else lines
lines = lines[args.id:args.id + 500000]
error_list = open('error.txt', 'a+')
for filename_idx, line in enumerate(lines):
    filename, person_id = line.split(',')
    if filename.find('FURTHER_00037') < 0:
        continue

    video_filename = os.path.join(args.video_direc, filename + '.mp4')
    landmarks_filename = os.path.join(args.landmark_direc, filename + '.pkl')
    dst_filename = os.path.join(args.save_direc, filename + '.npz')
    if os.path.getsize(dst_filename) == 0:
        print(dst_filename)
    #assert os.path.isfile(
    #    video_filename), f"File does not exist. Path input: {video_filename}"
    if not os.path.isfile(video_filename):
        print(f"File does not exist. Path input: {video_filename}, skip")
        continue
    assert os.path.isfile(
        landmarks_filename
    ), f"File does not exist. Path input: {landmarks_filename}"

    if os.path.exists(dst_filename) and os.path.getsize(dst_filename) > 0:
        #print(f"{dst_filename} exists, skip")
        continue

    print(f'idx: {filename_idx} \tProcessing.\t{filename}')
    # Extract mouth patches from segments
    sequence = dataloader.load_data(
        modality,
        video_filename,
        landmarks_filename,
    )

    try:
        if not os.path.exists(dst_filename) or os.path.getsize(
                dst_filename) == 0:
            save2npz(dst_filename, data=sequence)
    except:  # AssertionError:
        error_list.write(f'{video_filename}\n')
        print(f'!!!error {video_filename}, skip')
        #continue
