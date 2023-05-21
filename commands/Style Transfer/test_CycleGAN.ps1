Set-Location .\lib\CycleGAN

python .\test.py --dataroot ../../../data/cezanne2photo --name style_cezanne_pretrained --model test --no_dropout

Set-Location ..\..