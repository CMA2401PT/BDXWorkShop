import os
from canvas import Canvas
from canvas import irio
from artists.cmd_midi_music import Artist as CmdMusicArtist


# music_input = 'data/Weight_of_the_World_the_End_of_YoRHa.m4a'
midi_input = 'data/Undertale.mid'
canvas_output = 'output/sample07/midi_music_output.bdx'
os.makedirs('output/sample07', exist_ok=True)

canvas = Canvas()
p = canvas


artist = CmdMusicArtist(canvas=canvas, y=p.y+10)
music_cmds = artist.convert_music_to_mc_sound(midi_input, pling=True)

artist.write_to_cbs(music_cmds, x=0, y=0, z=0,
                    dir1=(1, 0, 0), dir1_lim=16,
                    dir2=(0, 0, 1), dir2_lim=16,
                    dir3=(0, 1, 0), cmds_wrapper="execute @a ~~~ playsound note.{} @s ~~~ {:.3f} {:.3f}")
artist.to_canvas()

final_ir = canvas.done()
irio.dump_ir_to_bdx(final_ir, canvas_output, need_sign=True, author='2401PT')
