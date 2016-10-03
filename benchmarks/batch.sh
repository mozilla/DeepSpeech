for batch in 1 2 4 8 16 32 64 $(seq 100 10 1600);
do
	python deepspeech.py --ldc93s1 $batch --run 3 --training-iters 200 --warpctc --dropout-rate 0.002 --csv warpctc_batch-${batch}_200.csv --plot  warpctc_batch-${batch}_200.png;
done;

python ../multiplot.py --csv batch/*.csv --plot batch/warpctc_titanx_batch_1-1600.png --plot-type time --title "Batch size influence" --x-label "Batch sizes (1, 2, 4, 8, 16, 32, 64, 100-1600)" --y-label "Exec. time (s)"
