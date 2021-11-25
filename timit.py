import glob
import os
import shutil


phn_file =glob.glob('./data/*.phn',recursive=True)
wav_file =glob.glob('./data/*.wav',recursive=True)
for i,v in enumerate(phn_file):
  w = wav_file[i]
  if i<(len(phn_file)*8)//10+1:
    shutil.move(v,os.path.join('../buckeye_directory/train',os.path.basename(v)))
    shutil.move(w,os.path.join('../buckeye_directory/train',os.path.basename(w)))
  #elif i<(len(phn_file)*9)//10+1:
  else:
    shutil.move(v,os.path.join('../buckeye_directory/test',os.path.basename(v)))
    shutil.move(w,os.path.join('../buckeye_directory/test',os.path.basename(w)))
#  else:
#    shutil.move(v,os.path.join('../SegFeat/timit_directory/val',os.path.basename(v).replace('PHN','phn')))
#    shutil.move(w,os.path.join('../SegFeat/timit_directory/train',os.path.basename(w)))
