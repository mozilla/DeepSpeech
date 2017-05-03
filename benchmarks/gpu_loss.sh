python deepspeech.py --ldc93s1 1400 --run 10 --training-iters 800 --ctc --dropout-rate 0.002 --csv time/ctc_titanx_1400.csv --plot ctc.png
python deepspeech.py --ldc93s1 1300 --run 10 --training-iters 800 --warpctc --dropout-rate 0.002 --csv time/warpctc_titanx_1300.csv --plot warpctc.png
python ../plot.py --csv time/ctc_titanx_1400.csv --plot time/ctc_loss_titanx_1400.png --plot-type loss
python ../plot.py --csv time/ctc_titanx_1400.csv --plot time/ctc_valerr_titanx_1400.png --plot-type valerr
python ../plot.py --csv time/warpctc_titanx_1300.csv --plot time/warpctc_loss_titanx_1300.png --plot-type loss
python ../plot.py --csv time/warpctc_titanx_1300.csv --plot time/warpctc_valerr_titanx_1300.png --plot-type valerr
