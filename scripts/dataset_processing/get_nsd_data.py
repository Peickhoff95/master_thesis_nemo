# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# USAGE: python get_nsd_data.py --data_root=<where to put data>
#        --data_set=<datasets_to_download> --num_workers=<number of parallel workers>

import argparse
import fnmatch
import functools
import json
import logging
import multiprocessing
import os
import subprocess
import zipfile
import urllib.request

from sox import Transformer
from tqdm import tqdm

import ipdb

parser = argparse.ArgumentParser(description="Noisy Speech Database Data download")
parser.add_argument("--data_root", required=True, default=None, type=str)
parser.add_argument("--data_sets", default="ALL", type=str)
parser.add_argument("--num_workers", default=4, type=int)
parser.add_argument("--log", dest="log", action="store_true", default=False)
args = parser.parse_args()


URLS = {
    "noisy_trainset_28spk_wav": ("https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip"),
    "noisy_trainset_56spk_wav": ("https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_56spk_wav.zip"),
    "clean_trainset_28spk_wav": ("https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip"),
    "clean_trainset_56spk_wav": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip",
    "trainset_28spk_txt": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/trainset_28spk_txt.zip",
    "trainset_56spk_txt": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/trainset_56spk_txt.zip",
    "noisy_testset_wav": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip",
    "clean_testset_wav": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip",
    "testset_txt": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/testset_txt.zip",
    "logfiles": "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/logfiles.zip",
}

def __retrieve_with_progress(source: str, filename: str):
    """
    Downloads source to destination
    Displays progress bar
    Args:
        source: url of resource
        destination: local filepath
    Returns:
    """
    with open(filename, "wb") as f:
        response = urllib.request.urlopen(source)
        total = response.length

        if total is None:
            f.write(response.content)
        else:
            with tqdm(total=total, unit="B", unit_scale=True, unit_divisor=1024) as pbar:
                for data in response:
                    f.write(data)
                    pbar.update(len(data))


def __maybe_download_file(destination: str, source: str):
    """
    Downloads source to destination if it doesn't exist.
    If exists, skips download
    Args:
        destination: local filepath
        source: url of resource
    Returns:
    """
    source = URLS[source]
    if not os.path.exists(destination):
        logging.info("{0} does not exist. Downloading ...".format(destination))

        __retrieve_with_progress(source, filename=destination + ".tmp")

        os.rename(destination + ".tmp", destination)
        logging.info("Downloaded {0}.".format(destination))
    else:
        logging.info("Destination {0} exists. Skipping.".format(destination))
    return destination


def __extract_file(filepath: str, data_dir: str):
    try:
        zip = zipfile.ZipFile(filepath)
        zip.extractall(data_dir)
        zip.close()
        if os.path.exists(os.path.join(filepath[:-4],"__MACOSX",)):
            os.rmdir(os.path.join(filepath[:-4],"__MACOSX",))
    except Exception:
        logging.info("Not extracting. Maybe already there?")


def __process_metadata(data, wav_path_input: str, wav_path_target: str, txt_path: str):
    """
    Captures metadata of wav files.
    Args:
        wav_path_input: path to noisy wav files
        wav_path_target: path to clean wav files
        txt_path: path to transcript text files
    Returns:
        a list of metadata entries for processed files.
    """
    entries = []
    file_name, noise_type, snr = data

    transcript_text = ""
    with open(os.path.join(txt_path, file_name + ".txt"), "r") as f:
        for line in f:
            transcript_text += line.strip()

    wav_file_input = os.path.join(wav_path_input, file_name + ".wav")
    wav_file_target = os.path.join(wav_path_target, file_name + ".wav")

    # check duration
    duration = subprocess.check_output("soxi -D {0}".format(wav_file_input), shell=True)

    entry = {}
    entry["input"] = os.path.abspath(wav_file_input)
    entry["target"] = os.path.abspath(wav_file_target)
    entry["duration"] = float(duration)
    entry["text"] = transcript_text
    entry["noise_type"] = noise_type
    entry["snr"] = snr.strip()
    entries.append(entry)

    return entries


def __process_data(data_folder_wav_input: str, data_folder_wav_target: str, data_folder_txt: str, log_file: str, manifest_file: str, num_workers: int):
    """
    Converts flac to wav and build manifests's json
    Args:
        data_folder_wav_input: source with noisy wav files
        data_folder_wav_target: source with clean wav files
        data_folder_txt: source with transcripts
        manifest_file: where to store manifest
        num_workers: number of parallel workers processing files
    Returns:
    """

    data = []
    entries = []

    with open(log_file, "r") as f:
        for line in f:
            filename, noise_type, snr = line.split(" ")
            data.append([filename, noise_type, snr])

    with multiprocessing.Pool(num_workers) as p:
        processing_func = functools.partial(__process_metadata, wav_path_input=data_folder_wav_input, wav_path_target=data_folder_wav_target, txt_path=data_folder_txt)
        results = p.imap(processing_func, data, chunksize=5)
        for result in tqdm(results, total=len(data)):
            entries.extend(result)

    with open(manifest_file, "w") as fout:
        for m in entries:
            fout.write(json.dumps(m) + "\n")


def main():
    data_root = args.data_root
    data_sets = args.data_sets
    num_workers = args.num_workers

    if args.log:
        logging.basicConfig(level=logging.INFO)

    if data_sets == "ALL":
        data_sets = "clean_testset_wav,clean_trainset_28spk_wav,clean_trainset_56spk_wav,logfiles,noisy_testset_wav,noisy_trainset_28spk_wav,noisy_trainset_56spk_wav,testset_txt,trainset_28spk_txt,trainset_56spk_txt"
        data_sets_wav_input = "noisy_testset_wav,noisy_trainset_28spk_wav,noisy_trainset_56spk_wav"
        data_sets_wav_target = "clean_testset_wav,clean_trainset_28spk_wav,clean_trainset_56spk_wav"
        data_sets_txt = "testset_txt,trainset_28spk_txt,trainset_56spk_txt"
        logfiles = "log_testset.txt,log_trainset_28spk.txt,log_trainset_56spk.txt"
    #if data_sets == "mini":
    #    data_sets = "dev_clean_2,train_clean_5"
    for data_set in data_sets.split(","):
        logging.info("\n\nWorking on: {0}".format(data_set))
        filepath = os.path.join(data_root, data_set + ".zip")
        logging.info("Getting {0}".format(data_set))
        __maybe_download_file(filepath, data_set)
        logging.info("Extracting {0}".format(data_set))
        __extract_file(filepath, data_root)

    for data_set_wav_input, data_set_wav_target, data_set_txt, logfile in zip(data_sets_wav_input.split(","), data_sets_wav_target.split(","), data_sets_txt.split(","), logfiles.split(",")):
        logging.info("Processing {0}".format(data_set_txt))
        __process_data(
            os.path.join(data_root, data_set_wav_input,),
            os.path.join(data_root, data_set_wav_target, ),
            os.path.join(data_root, data_set_txt, ),
            os.path.join(data_root, logfile, ),
            os.path.join(data_root, data_set_txt + ".json"),
            num_workers=num_workers,
        )

    logging.info("Done!")


if __name__ == "__main__":
    main()
