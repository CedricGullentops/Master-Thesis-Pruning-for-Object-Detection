**L2Prune:**
**Hard:**
python3 Pruning.py 2.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/l2prune-h-2.5
python3 Pruning.py 5.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/l2prune-h-5.0
python3 Pruning.py 7.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/l2prune-h-7.5
python3 Pruning.py 10.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/l2prune-h-10.0
python3 Pruning.py 12.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/l2prune-h-12.5
---
---
**GeometricMedian:**
**Hard:**
python3 Pruning.py 2.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/geometricmedian-h-2.5 -me geometricmedian
python3 Pruning.py 5.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/geometricmedian-h-5.0 -me geometricmedian
python3 Pruning.py 7.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/geometricmedian-h-7.5 -me geometricmedian
python3 Pruning.py 10.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/geometricmedian-h-10.0 -me geometricmedian
python3 Pruning.py 12.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 10000 -s backup/pruned/geometricmedian-h-12.5 -me geometricmedian
---
---
**Centripetal-SGD:**
**Hard**
python3 Pruning.py 5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 5000 -me centripetalSGD_even
python3 Pruning.py 7.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 5000 -me centripetalSGD_even
python3 Pruning.py 10 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 5000 -me centripetalSGD_even
python3 Pruning.py 12.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 5000 -me centripetalSGD_even
python3 Pruning.py 20 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 5000 -me centripetalSGD_even
python3 Pruning.py 40 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 5000 -me centripetalSGD_even

