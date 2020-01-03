# Code sample for finding dependencies
import lightnet as ln
import torch

model = ln.models.Yolo()

randomInput = torch.rand(1, 3, 416, 416)
traced_cell = torch.jit.trace(model, randomInput)
traced_cell_output = traced_cell.code

listed_trace = [s.strip() for s in traced_cell_output.splitlines()]

for text in listed_trace:
    print(text)
