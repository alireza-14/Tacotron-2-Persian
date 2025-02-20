
import argparse
import os
import shutil
import librosa
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from datasets import load_from_disk, load_dataset

from tac2persian.utils.generic import load_config
from tac2persian.utils.g2p.g2p import Grapheme2Phoneme
from tac2persian.utils.audio import log_melspectrogram, trim_silence


def normalize_text(text):
    if text[-1] not in [".", "!", "?"]:
        text = text + "."
    return text


def compute_features(source_audio_path, 
                     file_name, 
                     text, 
                     speaker_name, 
                     out_melspecs_path, 
                     g2p):
    try:
        text = normalize_text(text)
        out_mel_path = os.path.join(out_melspecs_path, file_name + ".npy")
        phoneme = g2p.text_to_phone(text, language="fa")
        phoneme_idx = g2p.phone_to_sequence(phoneme)
        phoneme_idx = ','.join(map(str, phoneme_idx))
        audio, _ = librosa.core.load(source_audio_path, sr=config["mel_params"]["sample_rate"])
        audio = trim_silence(audio, config["ref_level_db"])
        melspec = log_melspectrogram(audio, **config["mel_params"])
        np.save(out_mel_path, melspec)
        meta_line = f"{file_name}|{speaker_name}|{text}|{phoneme}|{melspec.shape[1]}|{phoneme_idx}"
        
        return meta_line
    except:
        print(f"Error in processing {file_name}")
        
        return None

def preprocess(dataset_path, output_path, target_speakers, config, num_workers, split):
    r"""Preprocesses audio files in the dataset."""
    
    try:
        dataset = load_from_disk(dataset_path)[split]
    except:
        dataset = load_dataset("mozilla-foundation/common_voice_13_0", "fa")[split]
    
    # Load G2P module
    g2p = Grapheme2Phoneme()

    dataset = dataset.filter(lambda example: example['client_id'] in target_speakers)

    speaker_name_map = {v:f"speaker_fa_{itr_spk}" for itr_spk, v in target_speakers}

    executor = ProcessPoolExecutor(max_workers=num_workers)
    
    # Create metafile and copy files
    metafile = []
    # Create final directory
    out_melspecs_path = os.path.join(output_path, "melspecs")
    os.makedirs(out_melspecs_path, exist_ok=True)
    for sample in dataset:
        # * change dataset meta file to new HF datasets.Dataset format
        speaker_name = speaker_name_map[sample['client_id']]
        file_name, text = os.path.basename(sample['path']), sample['sentence']
        file_name_ =  + "_" + file_name.split(".")[0]
        source_audio_path = sample['path']
        meta_line = executor.submit(partial(compute_features, 
                                            source_audio_path, 
                                            file_name_, 
                                            text, 
                                            speaker_name, 
                                            out_melspecs_path, 
                                            g2p))
        metafile.append(meta_line)

    metafile = [metaline.result() for metaline in metafile if metaline is not None]
    print(metafile)
    
    # Write metafile
    with open(os.path.join(output_path, "metadata.txt"), "w") as final_meta:
        for l in metafile:
            final_meta.write(l + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=5)
    args = parser.parse_args()
    target_speakers = []
    with open("static/selected_speakers.tsv", "r") as rfile:
        target_speakers = [i.strip() for i in rfile.readlines()]

    config = load_config(os.path.join(args.config_path, "config.yml"))
    preprocess(args.dataset_path, args.output_path, target_speakers, config, args.num_workers)