import os
from canvas import Canvas
from canvas import irio
from collections import defaultdict
from artists.map_art import Artist as MapArtArtist

image_input="data\QQ图片20220115231611.jpg"
output_dir= 'output/sample08'
os.makedirs(output_dir,exist_ok=True)

canvas = Canvas()
p = canvas
artist = MapArtArtist(canvas=canvas,y=0)

artist.add_img(img_path=image_input,
               level_x=2,level_y=2,  # 希望用几张地图实现呢?
               d3=False, #这个还没实现，先等等吧
               save_resized_file_to=os.path.join(output_dir,'resized_img.png'), #只是缩放了图片
               save_preview_to=os.path.join(output_dir,"preview.png"), # 效果预览图，不出意外导入游戏就是这样的
               )
artist.to_canvas()

final_ir = canvas.done()
irio.dump_ir_to_bdx(final_ir,
                    os.path.join("map.bdx"), # 保存文件路径
                    need_sign=True,
                    author='2401PT')
