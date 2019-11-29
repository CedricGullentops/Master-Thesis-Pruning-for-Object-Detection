#
#   Pruning process
#

# Basic imports
import subprocess

#proc = subprocess.Popen(["python3 bin/test.py backup/yolov2-640/final.pt -n cfg/yolov2-640.py -c"], stdout=subprocess.PIPE, shell=True)
#(out, err) = proc.communicate()
#print(out)

proc = subprocess.Popen(["python3 Pruning.py 5.0 backup/yolov2-640/final.pt -n cfg/yolov2-640.py -c -s ./pruned"], stdout=subprocess.PIPE, shell=True)
(out, err) = proc.communicate()

#proc = subprocess.Popen(["python3 bin/test.py backup/yolov2-640/final.pt -n cfg/yolov2-640.py -c"], stdout=subprocess.PIPE, shell=True)
#(out, err) = proc.communicate()
#print(out)

#proc = subprocess.Popen(["python3 bin/train.py ./pruned/pruned.pt -n ./cfg/test_cfg.py -pn ./pruned/params.state.pt -c"], stdout=subprocess.PIPE, shell=True)
#(out, err) = proc.communicate()
