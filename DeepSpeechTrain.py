#!/usr/bin/env python3
# Train deepspeech on Naomi audiolog data
import sqlite3
import numpy as np
import deepspeech
import wave
import jiwer
import os
from random import random
import shutil


if(__name__ == "__main__"):
    audiolog_dir = "/home/pi/.naomi/audiolog/"
    audiolog_db = "{}/audiolog.db".format(audiolog_dir)
    working_dir = "/home/pi/DeepSpeech_Adapt/voice/dsmodel"
    os.makedirs("{}/train".format(working_dir), exist_ok=True)
    with open("{}/train/train.csv".format(working_dir), "w") as f:
        f.write("wav_filename,wav_filesize,transcript\n")
    os.makedirs("{}/dev".format(working_dir), exist_ok=True)
    with open("{}/dev/dev.csv".format(working_dir), "w") as f:
        f.write("wav_filename,wav_filesize,transcript\n")
    os.makedirs("{}/test".format(working_dir), exist_ok=True)
    with open("{}/test/test.csv".format(working_dir), "w") as f:
        f.write("wav_filename,wav_filesize,transcript\n")
    # initialize deepspeech
    model_file_path = '/home/pi/deepspeech-0.6.1-models/output_graph.tflite'
    beam_width = 500
    model = deepspeech.Model(model_file_path, beam_width)
    lm_file_path = '/home/pi/deepspeech-0.6.1-models/lm.binary'
    trie_file_path = '/home/pi/deepspeech-0.6.1-models/trie'
    lm_alpha = 0.75
    lm_beta = 1.85
    model.enableDecoderWithLM(lm_file_path, trie_file_path, lm_alpha, lm_beta)
    # connect to database
    try:
        conn = sqlite3.connect(audiolog_db)
    except sqlite3.OperationalError:
        print("Can't connect to database")
        exit(1)
    c = conn.cursor()
    c.execute(
        " ".join([
            "select filename,verified_transcription",
            "from audiolog",
            "where type='active'",
            " and verified_transcription!=''"
            " and speaker=?"
        ]),
        ("Aaron",)
    )
    rows = c.fetchall()
    for row in rows:
        randcat = random()
        cat = "train"
        if(randcat > 0.70):
            cat = "dev"
        if(randcat > 0.90):
            cat = "test"
        (filename, verified_transcription) = row
        # full file path
        filepath_from = os.path.join(audiolog_dir, filename)
        filepath_to = os.path.join(working_dir, cat, filename)
        # Get the size of the file
        filesize = os.path.getsize(filepath_from)
        with open("{}/{}/{}.csv".format(working_dir, cat, cat), "a+") as f:
            f.write("{},{},{}\n".format(filename, filesize, verified_transcription.lower()))
        shutil.copyfile(filepath_from, filepath_to)
    conn.commit()
    conn.close()
