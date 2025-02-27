import cv2
import os
from argparse import ArgumentParser
from pathlib import Path


def arguments():
    parser = ArgumentParser(parents=[])

    parser.add_argument(
        "--image_dir",
        type=str,
        default="csldaily/sentences/frames_512x512",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="dataset_creation/csl-daily",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="S000000_P0000_T00",
    )

    params, unknown = parser.parse_known_args()
    return params


def images_to_video(image_dir, save_dir, name, fps=25):
    image_folder = f"{image_dir}/{name}"
    video_name = name
    tmp_dir = "/tmp"

    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    image_files = [
        f for f in os.listdir(image_folder) if f.endswith(".jpg") or f.endswith(".png")
    ]
    image_files.sort()

    image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(image_path)
    height, width, channels = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for AVI format
    out = cv2.VideoWriter(f"{tmp_dir}/{video_name}.mp4", fourcc, fps, (width, height))

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        out.write(image)

    out.release()

    cp_cmd = "cp {} {}"
    # copy video to tmp
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    os.system(
        cp_cmd.format(f"{tmp_dir}/{video_name}.mp4", f"{save_dir}/{video_name}.mp4")
    )


if __name__ == "__main__":

    args = arguments()
    image_dir = args.image_dir
    save_dir = args.save_dir
    name = args.name
    images_to_video(image_dir, save_dir, name)
