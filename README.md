## COCO Dataset : PythonAPI
ich nutze die gleiche EC2 wie für Detectron:
Deep Learning AMI Ubuntu Linux - 2.5_Jan2018 (ami-01148d6e); Instance Type: p2.xlarge.

Habe vorher andere EC2-Instanzen probiert (ohne GPU). Dort war Red Hat installiert und es gab Probleme, um cocoapi/PythonAPI zum Laufen zu bekommen,
weil bestimmte Python Module wie python-tk sich nicht installieren ließen.
Auf Deep Learning AMI Ubuntu Linux - 2.5_Jan2018 (ami-01148d6e) muss man eigentlich keine Installation von cocoapi/PythonAPI mehr machen, wenn
das mit [`setup.sh`](https://github.com/lborke/Detectron/blob/master/setup.sh) bereits gemacht wurde.
In Zukunft sollte man cocoapi/PythonAPI auf einer billigeren EC2 ohne GPU unter Ubuntu installieren.
cocoapi/PythonAPI läuft stand-alone, also ohne Caffe und Detectron.

Der Inhalt vom [coco Ordner](https://github.com/lborke/Detectron/tree/master/coco) ist eine Kopie von meinem Ordner `/home/ubuntu/coco` auf der EC2.
Die Source-Dateien liegen in `/home/ubuntu/src/cocoapi`, und werden nach 'make install' nicht mehr benötigt,
da daraus eine selbständige Python-Bibliothek erstellt wird, nämlich `from pycocotools.coco import COCO`.


## Start jupyter notebook via Browser/Tunnel

Wenn man den [coco Ordner](https://github.com/lborke/Detectron/tree/master/coco) nach `/home/ubuntu/coco` auf der EC2 kopiert (am besten mit WinSCP),
und eine PuTTY-Verbindung mit der EC2 hergestellt hat, wird mit den folgenden Shell-Kommandos `jupyter notebook` gestartet.

```
cd /home/ubuntu/coco
nohup jupyter notebook &
tail nohup.out
```

Die letzte Zeile `tail nohup.out` zeigt den Inhalt von `nohup.out` und man kann daraus den URL-Link für seinen lokalen Browser entnehmen,
um `jupyter notebook` interaktiv zu nutzen, z.B.:

```
to login with a token:
    http://localhost:8888/?token=94218f778c6971a0f13f4acde7ca15114f2bcbd8f00c6371
```

Damit das funktioniert, muss man in der PuTTY-Configuration unter Connection/SSH/Tunnels `localhost:8888` hinzufügen,
der Rest ist ganz gewöhnlich wie bei einer normalen SSH-Session via PuTTY.


## PythonAPI ohne Display-Output
Sofern man keine Bilder/Outputs erstellt, lassen sich die meisten Sachen/Analysen in der Kommandozeile in Python ausführen, z.B.:

```python

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

dataDir='.'
dataType='val2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))


# get all images containing given categories, select one at random
catIds = coco.getCatIds(catNms=['person','dog','skateboard'])
imgIds = coco.getImgIds(catIds=catIds)
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

# take a random image from the whole set (5000 in 'val2017')
imgIds = coco.getImgIds()
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]

```

 ## Cocoapi for Windows
 [https://github.com/philferriere/cocoapi](https://github.com/philferriere/cocoapi)
