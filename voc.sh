#!/bin/sh

Copy the following script command.
start = 'date + %s'

echo "Prepare to download train-val2014 anotation zip file..."

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
unzip VOCtest_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

