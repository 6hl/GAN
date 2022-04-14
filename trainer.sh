#!/bin/bash
python main.py -m DCGAN -e 100 -lr 0.0001
python main.py -m WGAN -i 20000 -lr 0.00005 -b 64
python main.py -m ACGAN -e 800 -lr 0.0003 -b 64