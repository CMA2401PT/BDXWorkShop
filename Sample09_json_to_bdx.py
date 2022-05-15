import json
import os 
from canvas import irio
from canvas import Canvas

import_json_file = 'data/六分之一的小岛.json'
canvas_output = 'output/sample09/六分之一的小岛.bdx'
os.makedirs('output/sample09', exist_ok=True)

canvas = Canvas()
p = canvas
with open(import_json_file,"r") as f:
    inBlocks=json.load(f)
    print(f"{len(inBlocks)} blocks in total")
    for block in inBlocks:
        x,y,z=block["x"],block["y"],block["z"]
        name=block["name"][10:]
        data=block["aux"]
        canvas.setblock(name, data, x,y,z)
final_ir = canvas.done()
irio.dump_ir_to_bdx(final_ir, canvas_output, need_sign=True, author='2401PT')