#! /bin/bash
#copies all of frames and video datasets that rajat loaded into his folder

printf " activitynet frames "
cp -r /home/rmodi/to_share/david/MA-LMM/data/activitynet/frames ./data/activitynet
printf" activitynet vids "
cp -r /home/rmodi/to_share/david/MA-LMM/data/activitynet/videos ./data/activitynet
printf" breakfast frames "
cp -r /home/rmodi/to_share/david/MA-LMM/data/breakfast/frames ./data/breakfast
printf" breakfast vids "
cp -r /home/rmodi/to_share/david/MA-LMM/data/breakfast/videos ./data/breakfast
printf" coin frames "
cp -r /home/rmodi/to_share/david/MA-LMM/data/coin/frames ./data/coin
printf" coin vids "
cp -r /home/rmodi/to_share/david/MA-LMM/data/coin/videos ./data/coin
printf" lvu frames "
cp -r /home/rmodi/to_share/david/MA-LMM/data/lvu/frames ./data/lvu
printf"lvu vids "
cp -r /home/rmodi/to_share/david/MA-LMM/data/lvu/videos ./data/lvu
printf" msvrtt frames "
cp -r /home/rmodi/to_share/david/MA-LMM/data/msrvtt/frames ./data/msrvtt
printf" msvrtt vids "
cp -r /home/rmodi/to_share/david/MA-LMM/data/msrvtt/videos ./data/msrvtt
printf" msvd frames "
cp -r /home/rmodi/to_share/david/MA-LMM/data/msvd/frames ./data/msvd
printf" msvd vids "
cp -r /home/rmodi/to_share/david/MA-LMM/data/msvd/videos ./data/msvd
printf" youcook frames "
cp -r /home/rmodi/to_share/david/MA-LMM/data/youcook2/frames ./data/youcook2
printf" youcook vids "
cp -r /home/rmodi/to_share/david/MA-LMM/data/youcook2/videos ./data/youcook2
printf"done"