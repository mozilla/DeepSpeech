import glob
import os
import tarfile
import pandas as pd

from deepspeech_training.util.importers import get_importers_parser

COLUMN_NAMES = ["wav_filename", "wav_filesize", "transcript"]

def extract(archive_path, target_dir):
    print(f"Extracting {archive_path} into {target_dir}...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(target_dir)

def preprocess_data(tgz_file, target_dir):
    # First extract main archive and sub-archives
    extract(tgz_file, target_dir)
    main_folder = os.path.join(target_dir, "data_aishell")

    wav_archives_folder = os.path.join(main_folder, "wav")
    for targz in glob.glob(os.path.join(wav_archives_folder, "*.tar.gz")):
        extract(targz, main_folder)

    transcripts_path = os.path.join(main_folder, "transcript", "aishell_transcript_v0.8.txt")
    with open(transcripts_path) as fin:
        transcripts = dict((line.split(" ", maxsplit=1) for line in fin))

    def load_set(glob_path):
        set_files = []
        for wav in glob.glob(glob_path):
            try:
                wav_filename = wav
                wav_filesize = os.path.getsize(wav)
                transcript_key = os.path.splitext(os.path.basename(wav))[0]
                transcript = transcripts.get(transcript_key, "").strip("\n")
                set_files.append((wav_filename, wav_filesize, transcript))
            except KeyError:
                print(f"Warning: Missing transcript for WAV file {wav}.")
        return set_files

    for subset in ["train", "dev", "test"]:
        print(f"Loading {subset} set samples...")
        subset_files = load_set(os.path.join(main_folder, subset, "S*", "*.wav"))
        df = pd.DataFrame(data=subset_files, columns=COLUMN_NAMES)

        if subset == "train":
            durations = (df["wav_filesize"] - 44) / 16000 / 2
            df = df[durations <= 10.0]
            print(f"Trimming {subset} samples > 10 seconds: {(durations > 10.0).sum()}")

        dest_csv = os.path.join(target_dir, f"aishell_{subset}.csv")
        print(f"Saving {subset} set into {dest_csv}...")
        df.to_csv(dest_csv, index=False)

def main():
    parser = get_importers_parser(description="Import AISHELL corpus")
    parser.add_argument("aishell_tgz_file", help="Path to data_aishell.tgz")
    parser.add_argument(
        "--target_dir",
        default="",
        help="Target folder to extract files into and put the resulting CSVs. Defaults to the same folder as the main archive.",
    )
    params = parser.parse_args()

    if not params.target_dir:
        params.target_dir = os.path.dirname(params.aishell_tgz_file)

    preprocess_data(params.aishell_tgz_file, params.target_dir)

if __name__ == "__main__":
    main()

