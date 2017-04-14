#!/bin/bash
  
root_dir="$HOME/data/KITTI/" #your path to kitti dataset
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
label_dir="label_2car"                #path to labels of car
name="image_2"
for dataset in training testing
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi

  echo "Create list for $name $dataset..."
  dataset_file=./main/$dataset.txt

  img_file=$bash_dir/$dataset"_img.txt"
  cp $dataset_file $img_file
  sed -i "s/^/training\/$name\//g" $img_file
  sed -i "s/$/.png/g" $img_file

  label_file=$bash_dir/$dataset"_label.txt"
  cp $dataset_file $label_file
  sed -i "s/^/training\/$label_dir\/xml\//g" $label_file
  sed -i "s/$/.xml/g" $label_file

  paste -d' ' $img_file $label_file >> $dst_file

  rm -f $label_file
  rm -f $img_file

  # Generate image name and size infomation.
  if [ "$dataset" == "testing" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "training" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
