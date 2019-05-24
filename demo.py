#encoding=utf8
from ctpnport import *
from crnnport import *
import time

#ctpn
text_detector = ctpnSource()
#crnn
model,converter = crnnSource()
from PIL import Image

pics = os.listdir("./img/")
print pics
for im_name in pics:
    im_path = "./img/" + im_name
    im = cv2.imread(im_path)
    if im is None:
      continue
    st = time.time()
    img,text_recs,img_with_box = getCharBlock(text_detector,im)
    print "Time:", time.time() - st
    st = time.time()
    text_res = crnnRec(model,converter,img,text_recs)
    print "Time:", time.time() - st
    img_with_box = cv2.cvtColor(img_with_box,cv2.COLOR_BGR2RGB)
    I = Image.fromarray(img_with_box)
    I.save('./res/'+im_name)
    f = open('./res/'+im_name+'.txt','wb')
    f.write('\n'.join(text_res))
    f.close()
