#!/bin/sh

set -xe

ldc93s1_dir=`cd data/smoke_test; pwd`
ldc93s1_csv="${ldc93s1_dir}/LDC93S1.csv"
ldc93s1_wav="${ldc93s1_dir}/LDC93S1.wav"
ldc93s1_overlay_csv="${ldc93s1_dir}/LDC93S1_overlay.csv"
ldc93s1_overlay_wav="${ldc93s1_dir}/LDC93S1_reversed.wav"

play="python bin/play.py --number 1 --quiet"
compare="python bin/compare_samples.py --no-success-output"

if [ ! -f "${ldc93s1_csv}" ]; then
    echo "Downloading and preprocessing LDC93S1 example data, saving in ${ldc93s1_dir}."
    python -u bin/import_ldc93s1.py ${ldc93s1_dir}
fi;

if [ ! -f "${ldc93s1_overlay_csv}" ]; then
    echo "Reversing ${ldc93s1_wav} to ${ldc93s1_overlay_wav}."
    sox "${ldc93s1_wav}" "${ldc93s1_overlay_wav}" reverse

    echo "Creating ${ldc93s1_overlay_csv}."
    printf "wav_filename\n${ldc93s1_overlay_wav}" > "${ldc93s1_overlay_csv}"
fi;

if ! $compare --if-differ "${ldc93s1_wav}" "${ldc93s1_overlay_wav}"; then
  echo "Sample comparison tool not working correctly"
  exit 1
fi

$play ${ldc93s1_wav} --augment overlay[source="${ldc93s1_overlay_csv}",snr=20] --pipe >/tmp/overlay-test.wav
if ! $compare --if-differ "${ldc93s1_wav}" /tmp/overlay-test.wav; then
  echo "Overlay augmentation had no effect or changed basic sample properties"
  exit 1
fi

$play ${ldc93s1_wav} --augment reverb[delay=50.0,decay=2.0] --pipe >/tmp/reverb-test.wav
if ! $compare --if-differ "${ldc93s1_wav}" /tmp/reverb-test.wav; then
  echo "Reverb augmentation had no effect or changed basic sample properties"
  exit 1
fi

$play ${ldc93s1_wav} --augment resample[rate=4000] --pipe >/tmp/resample-test.wav
if ! $compare --if-differ "${ldc93s1_wav}" /tmp/resample-test.wav; then
  echo "Resample augmentation had no effect or changed basic sample properties"
  exit 1
fi

$play ${ldc93s1_wav} --augment codec[bitrate=4000] --pipe >/tmp/codec-test.wav
if ! $compare --if-differ "${ldc93s1_wav}" /tmp/codec-test.wav; then
  echo "Codec augmentation had no effect or changed basic sample properties"
  exit 1
fi

$play ${ldc93s1_wav} --augment volume --pipe >/tmp/volume-test.wav
if ! $compare --if-differ "${ldc93s1_wav}" /tmp/volume-test.wav; then
  echo "Volume augmentation had no effect or changed basic sample properties"
  exit 1
fi
