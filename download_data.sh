#!/bin/sh

Copy the following script command.
start = 'date + %s'

echo "Prepare to download train-val2017 anotation zip file..."
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip annotations_trainval2017.zip
rm -f annotations_trainval2017.zip

echo "Prepare to download val2014 image zip file..."
wget -c http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm -f val2014.zip

echo "Prepare to download VOCtrainval_06-Nov-2007 zip file..."

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget -c http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
unzip VOCtest_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar

end = 'date + %s'
runtime = $((end - start))

echo "Download completed in " $runtime  " second"

"""'./data/datasets/coco/annotations/instances_val2014.json'
"""