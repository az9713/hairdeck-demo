"""python sheet.py OUT.png COLS FILE... -> one contact sheet of equal-sized images."""
import sys
from PIL import Image
out,cols,files=sys.argv[1],int(sys.argv[2]),sys.argv[3:]
ims=[Image.open(f).convert('RGB') for f in files]
w,h=ims[0].size
s=Image.new('RGB',(w*cols,h*((len(ims)+cols-1)//cols)),'white')
for i,m in enumerate(ims): s.paste(m,((i%cols)*w,(i//cols)*h))
s.save(out)
