python deepspeech.py --ldc93s1 350 --run 10 --training-iters 800 --warpctc --dropout-rate 0.002 --csv warpctc_titanx_350.csv --plot warpctc_350.png
python deepspeech.py --ldc93s1 350 --run 10 --training-iters 800 --warpctc --dropout-rate 0.002 --csv warpctc_gtx970_350.csv --plot warpctc_350.png
python ../plot.py --csv time/warpctc_titanx_350.csv --plot time/warpctc_loss_titanx_350.png --plot-type loss
python ../plot.py --csv time/warpctc_gtx970_350.csv --plot time/warpctc_loss_gtx970_350.png --plot-type loss
python ../plot.py --csv time/warpctc_titanx_350.csv --plot time/warpctc_valerr_titanx_350.png --plot-type valerr
python ../plot.py --csv time/warpctc_gtx970_350.csv --plot time/warpctc_valerr_gtx970_350.png --plot-type valerr
