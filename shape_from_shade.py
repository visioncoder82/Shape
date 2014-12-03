'''
Created on 1 Dec 2014

@author: narayan

'''

from os import listdir
import numpy as np
from PIL import Image
from pylab import *
import urllib, cStringIO
from scipy.ndimage import filters
from mpl_toolkits.mplot3d import axes3d


def Tsai_Shah_Shading(img, ps, qs, trails):
    shape = img.shape
    
    Zn  = np.zeros(shape)
    Zn1 = np.zeros(shape)
    
    Si1 = 0.01* np.ones(shape)
    Wn  =0.0001*0.0001
    
    p = np.zeros(shape)
    q = np.zeros(shape)
    
    Ps = ps* np.ones(shape)
    Qs = qs* np.ones(shape)
    
    for i in range(trails):
        p[2:shape[0]-1,2:shape[1]-1] = Zn1[2:shape[0]-1,2:shape[1]-1] - Zn1[2:shape[0]-1,1:shape[1]-2]
        q[2:shape[0]-1,2:shape[1]-1] = Zn1[2:shape[0]-1,2:shape[1]-1] - Zn1[1:shape[0]-2,2:shape[1]-1]
        
        pq  = 1.0 + p*p + q*q
        PQs = 1.0 + Ps*Ps + Qs*Qs
        
        fZ  = -1.0*(img - np.maximum(np.zeros(shape),(1+p*Ps+q*Qs)/(np.sqrt(pq)*np.sqrt(PQs))))
        dfZ = -1.0*((Ps+Qs)/(np.sqrt(pq)*np.sqrt(PQs))-(p+q)*(1.0+p*Ps+q*Qs)/(np.sqrt(pq*pq*pq)*np.sqrt(PQs)))
        Y   =  fZ + dfZ*Zn1
        K   =  Si1 * dfZ/(Wn+dfZ*Si1*dfZ)
        Si1 = (1.0 - K*dfZ)*Si1
        Zn  = Zn1 + K*(Y-dfZ*Zn1)
        Zn1 = Zn
        
        
    

        
    return Zn
       
     


URL = 'http://upload.wikimedia.org/wikipedia/commons/5/5f/Mysore_palace.jpg'
#URL = 'http://www.indiapicks.com/stamps/Gallery/1987-88/1276_White_Tiger.jpg'
#URL = 'http://upload.wikimedia.org/wikipedia/en/2/24/Lenna.png'

#Very Good Example 
#URL = 'http://www.3dcadbrowser.com/th/1/4/4404.jpg'

file = cStringIO.StringIO(urllib.urlopen(URL).read())
mysp = Image.open(file)
im = rgb2gray(np.array(mysp))

trials = 20
Z = Tsai_Shah_Shading(im, 0.01, 0.01, trials)
shape = Z.shape

display=plt.imshow(Z[2:shape[0]-1,2:shape[1]-1])
plt.xticks([]), plt.yticks([])
display.set_data(Z[2:shape[0]-1,2:shape[1]-1])
plt.show()

x = range(shape[0])
y = range(shape[1])

xx,yy = np.meshgrid(x,y)

print xx.shape, yy.shape, np.transpose(Z).shape
#h = plt.contourf(np.transpose(xx),np.transpose(yy),Z)
#plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(xx, yy, np.transpose(Z),rstride=50, cstride=50)
plt.show()




# figure()
# gray()
# F = Image.fromarray(im)
# F.show()

