"""python finish.py: title PNG, 90 s music edit, final encode -> out/regrowth.mp4."""
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FF = os.environ.get('FFMPEG', 'ffmpeg')
out = os.path.join(ROOT, 'out')


def text(path, s, size, y, font):
    im = Image.new('RGBA', (1920, 1080), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, size)
    w = d.textlength(s, font=f)
    d.text((1580 - w / 2, y), s, font=f, fill=(58, 40, 44, 255))
    im.save(path)


text(os.path.join(out, 'title.png'), 'Regrowth', 96, 330, r'C:\Windows\Fonts\georgia.ttf')
text(os.path.join(out, 'credit.png'), 'Hair made with HairDeck', 34, 460, r'C:\Windows\Fonts\georgia.ttf')

# Music: track b is 158 s and builds up from a quiet part at 20-35 s (loudness measured per 5 s). The edit holds that quiet
# part for the bald shots, then uses the build from 35 s for the regrowth. Film time of each part: 0-35, 33-48, 46-61, 59-90.
music = os.path.join(ROOT, 'assets', 'music', 'b.mp3')
fc = ('[0:a]atrim=0:35,asetpts=N/SR/TB[s1];'
      '[0:a]atrim=20:35,asetpts=N/SR/TB,volume=0.7[s2];'
      '[0:a]atrim=20:35,asetpts=N/SR/TB,volume=0.45[s3];'
      '[0:a]atrim=35:66,asetpts=N/SR/TB,volume=\'min(0.45+0.55*t/12,1)\':eval=frame[s4];'
      '[s1][s2]acrossfade=d=2[m1];[m1][s3]acrossfade=d=2[m2];[m2][s4]acrossfade=d=2,afade=t=in:d=1.5,afade=t=out:st=85.5:d=4.5,atrim=0:90,volume=-1dB[a]')
audio = os.path.join(out, 'music.m4a')
subprocess.run([FF, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', music, '-filter_complex', fc, '-map', '[a]', '-c:a', 'aac', '-b:a', '192k', audio], check=True)

fv = ('[0:v]fade=t=in:st=0:d=1.2,fade=t=out:st=88.6:d=1.4[b];'
      '[1:v]format=rgba,fade=t=in:st=84:d=1.5:alpha=1,fade=t=out:st=88.2:d=1:alpha=1[t];'
      '[2:v]format=rgba,fade=t=in:st=85.5:d=1.5:alpha=1,fade=t=out:st=88.2:d=1:alpha=1[c];'
      '[b][t]overlay=shortest=1[x];[x][c]overlay=shortest=1,format=yuv420p[v]')
subprocess.run([FF, '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-framerate', '24', '-i', os.path.join(out, 'frames', 'f%05d.png'),
                '-loop', '1', '-framerate', '24', '-t', '90', '-i', os.path.join(out, 'title.png'),
                '-loop', '1', '-framerate', '24', '-t', '90', '-i', os.path.join(out, 'credit.png'),
                '-i', audio, '-filter_complex', fv, '-map', '[v]', '-map', '3:a', '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
                '-c:a', 'copy', '-movflags', '+faststart', '-shortest', os.path.join(out, 'regrowth.mp4')], check=True)
print('done', os.path.getsize(os.path.join(out, 'regrowth.mp4')))
