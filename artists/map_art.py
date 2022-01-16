from PIL import Image
import numpy as np 
import einops
from numba import jit
from canvas import Canvas

cmap=[
    ["concrete 0", [220, 220, 220]], 
    ["concrete 8", [132, 132, 132]], 
    ["concrete 7", [65, 65, 65]], 
    ["concrete 15", [22, 22, 22]], 
    ["concrete 12", [88, 65, 44]], 
    ["concrete 14", [132, 44, 44]], 
    ["concrete 1", [186, 108, 44]], 
    ["concrete 4", [198, 198, 44]], 
    ["concrete 5", [108, 176, 22]], 
    ["concrete 13", [88, 108, 44]], 
    ["concrete 9", [66, 108, 132]], 
    ["concrete 3", [88, 132, 186]], 
    ["concrete 11", [44, 66, 152]], 
    ["concrete 10", [108, 55, 152]], 
    ["concrete 2", [152, 66, 186]], 
    ["concrete 6", [208, 108, 142]], 
    ["planks 0", [124, 100, 60]], 
    ["planks 1", [112, 74, 42]], 
    ["crimson_planks 0", [128, 54, 84]], 
    ["warped_planks 0", [50, 122, 120]], 
    ["dirt 1", [130, 94, 66]], 
    ["sandstone 0", [212, 200, 140]], 
    ["clay 0", [140, 144, 158]], 
    ["stone 0", [96, 96, 96]], 
    ["gold_ore 0", [86, 86, 86]], 
    ["netherrack 0", [96, 0, 0]], 
    ["quartz_block 0", [220, 216, 210]], 
    ["stained_hardened_clay 8", [116, 92, 84]], 
    ["warped_nylium 0", [18, 108, 116]], 
    ["sweet_berry_bush 0", [0, 108, 0]], 
    ["leaves 12", [55, 80, 20]], 
    ["leaves 14", [60, 78, 38]], 
    ["leaves 13", [45, 70, 45]], 
    ["glow_lichen 0", [108, 144, 128]], 
    ["crimson_hyphae 0", [80, 20, 25]], 
    ["warped_hyphae 0", [75, 37, 52]], 
    ["crimson_nylium 0", [162, 42, 42]], 
    ["warped_wart_block 0", [15, 155, 115]], 
    ["diamond_block 0", [78, 188, 182]], 
    ["iron_block 0", [144, 144, 144]], 
    ["redstone_block 0", [220, 0, 0]], 
    ["gold_block 0", [215, 205, 65]], 
    ["emerald_block 0", [0, 188, 50]], 
    ["lapis_block 0", [64, 110, 220]], 
    ["stained_hardened_clay 0", [180, 150, 140]], 
    ["stained_hardened_clay 7", [50, 35, 30]], 
    ["stained_hardened_clay 12", [65, 42, 30]], 
    ["slime 0", [108, 152, 48]], 
    ["web 0", [170, 170, 170]], 
    ["blue_ice 0", [138, 138, 220]], 
    ["grass 0", [125, 160, 75]]
]
def light_pixel(rgb):
    return [round(255/220*rgb[0]),255/220*rgb[1],255/220*rgb[2]]
def darkPixel(rgb):
    return [round(180/220*rgb[0]),180/220*rgb[1],180/220*rgb[2]]

color_list=[c[1] for c in cmap]
color_list=np.array(color_list).astype(np.float32)*(255/220)

color_list_d3=[]
block_names=[]
for block_def,rgb in cmap:
    color_list_d3.append(rgb)
    color_list_d3.append(darkPixel(rgb))
    color_list_d3.append(light_pixel(rgb))
    block_name,val=block_def.split(" ")
    block_names.append((block_name,int(val)))
color_list_d3=np.array(color_list_d3).astype(np.float32)
# block_name=[c[0] for c in cmap]

    
    

@jit(nopython=True) 
def closest_color(target_rgb,color_list):
    delta=np.inf
    best_match=None
    target_rgb=np.clip(target_rgb,0,255)
    tr,tg,tb=target_rgb
    for i,(r,g,b) in enumerate(color_list):
        if tr+r>256:
            d=2*((tr-r)**2)+4*((tg-g)**2)+3*((tb-b)**2)
        else:
            d=3*((tr-r)**2)+4*((tg-g)**2)+2*((tb-b)**2)
        if d<delta:
            delta=d
            best_match=(i,color_list[i])
    return best_match

@jit(nopython=True) 
def convert_palette(color_list,data:np.ndarray):
    converted_data=data.copy().astype(np.float32)
    block_map=np.zeros(data.shape[:2],dtype=np.uint8)
    flatten_data=converted_data.reshape((-1,3))
    flatten_block_map=block_map.reshape((-1,))
    converted_data.reshape((-1,3))
    h,w,_=converted_data.shape
    for pi,rgb in enumerate(flatten_data):
        mi,mrgb = closest_color(rgb,color_list)
        # (r,g,b),(mr,mg,mb)=rgb,mrgb
        drgb=rgb-mrgb
        if (pi+1)%w:
            flatten_data[pi+1]+=(7/16)*drgb
        elif pi<(h-1)*w:
            pj=pi+w
            flatten_data[pj]+=(5/16)*drgb
            if (pi+1)%w:
                pj=pi+w+1
                flatten_data[pj]+=(1/16)*drgb
            if pi%w:
                pj=pi+w-1
                flatten_data[pj]+=(3/16)*drgb
        flatten_data[pi]=mrgb
        flatten_block_map[pi]=mi
    converted_data=flatten_data.reshape((h,w,3)).astype(np.uint8)
    block_map=flatten_block_map.reshape((h,w))
    return (block_map,converted_data)

def convert_img(img:Image,level_x=2,level_y=2,d3=False):
    resized_img=img.resize((level_x*128,level_y*128))
    resized_img_np=np.array(resized_img)
    block_map,converted_img=convert_palette(color_list_d3 if d3 else color_list,resized_img_np)
    converted_img=Image.fromarray(converted_img)
    return block_map,converted_img,resized_img


# @jit(nopython=True)
def write_blocks(canvas:Canvas,block_map:np.ndarray):
    h,w=block_map.shape
    for r in range(h):
        for c in range(w):
            name,val=block_names[block_map[r,c]]
            canvas.setblock(name,val,r,0,c)
            if name =="sweet_berry_bush":
                canvas.setblock("grass",0,c,-1,r)
            if name =="leaves":
                canvas.setblock("grass",0,c,-1,r)

# to bdx, it's trivial
class Artist(Canvas):
    def __init__(self, canvas: Canvas, x=None, y=None, z=None):
        super().__init__()
        self.target_canvas = canvas
        self.target_xyz = {'ox': x, 'oy': y, 'oz': z}
        self.block_names=block_names

    def to_canvas(self):
        self_host_ir = self.done()
        self.target_canvas.load_ir(self_host_ir, merge=True, **self.target_xyz)
        return self
    

    
    def add_img(self,img_path:str,level_x=1,level_y=1,d3=False,save_resized_file_to=None,save_preview_to=None):
        img=Image.open(img_path)
        if not d3:
            block_map,converted_img,resized_img=convert_img(img,level_x,level_y,d3=False)
            if save_resized_file_to is not None:
                resized_img.save(save_resized_file_to)
            if save_preview_to is not None:
                converted_img.save(save_preview_to)
            write_blocks(self,block_map)
        else:
            raise NotImplemented("3D 过两天再实现")