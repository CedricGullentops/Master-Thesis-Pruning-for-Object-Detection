**L2Prune:**
**Hard:**
python3 Pruning.py 2.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-h-2.5
python3 Pruning.py 5.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-h-5.0
python3 Pruning.py 7.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-h-7.5
python3 Pruning.py 10.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-h-10.0
python3 Pruning.py 12.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-h-12.5
---
**Soft:**
python3 Pruning.py 2.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-s-2.5 -m soft
python3 Pruning.py 5.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-s-5.0 -m soft
python3 Pruning.py 7.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-s-7.5 -m soft
python3 Pruning.py 10.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-s-10.0 -m soft
python3 Pruning.py 12.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/l2prune-s-12.5 -m soft
---
---
**GeometricMedian:**
**Hard:**
python3 Pruning.py 2.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-h-2.5 -me geometricmedian
python3 Pruning.py 5.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-h-5.0 -me geometricmedian
python3 Pruning.py 7.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-h-7.5 -me geometricmedian
python3 Pruning.py 10.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-h-10.0 -me geometricmedian
python3 Pruning.py 12.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-h-12.5 -me geometricmedian
---
**Soft:**
python3 Pruning.py 2.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-s-2.5 -m soft -me geometricmedian
python3 Pruning.py 5.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-s-5.0 -m soft -me geometricmedian
python3 Pruning.py 7.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-s-7.5 -m soft -me geometricmedian
python3 Pruning.py 10.0 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-s-10.0 -m soft -me geometricmedian
python3 Pruning.py 12.5 backup/yolov2-416/final.pt -n cfg/yolov2-416.py -c -mb 15000 -s backup/pruned/geometricmedian-s-12.5 -m soft -me geometricmedian


