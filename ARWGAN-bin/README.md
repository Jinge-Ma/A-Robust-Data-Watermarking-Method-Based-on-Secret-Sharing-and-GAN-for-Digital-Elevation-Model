## Train
If you need to train ARWGAN-bin from scratch, you should use commond line as following.

      python mian.py new -n name -d data-dir -b batch-size -e epochs  -n noise
      
Environmental requirements:
+ Python == 3.7.4; Torch == 1.12.1 + cu102; Torchvision == 0.13.1; PIL == 7.2.0

## Test
Put the pre-trained model into pretrain floder, and you can test ARWGAN by command line as following.

      python test.py -o XXXX.pickle -c XXXX.pyt -s data-dir -n noise

## Acknowledgement
The codes are designed based on [ARWGAN](https://github.com/river-huang/ARWGAN).
