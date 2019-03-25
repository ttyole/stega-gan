import models.utility.stc.STC as stc
import numpy as np
import matplotlib.pyplot as plt
import random
from math import pi, sin
 
def str2bits(s):
    return list(map(lambda c: int(c),list(''.join('{:08b}'.format(b) for b in s.encode('utf8')))))

def bits2str(b):
    return list(bytes(int(''.join(map(lambda x: str(x),b))[x:x+8],2) for x in range(0,len(b),8)).decode("utf8"))
    


n = 512

prob_map = np.array([[abs(sin(x/(51.2 *pi))/2) for x in range(n)] for _ in range(n)])
cover = np.random.randint(0,256,(n,n))
message = "mon message cache"

c_cover = list(np.reshape(cover,np.size(cover)))
c_prob = list(np.reshape(prob_map,np.size(prob_map)))
# c_mess = str2bits(message)
c_mess = list(np.random.randint(0,2,100))

print(message)
print(c_mess)

(success,c_stego,c_lsb) = stc.embed(c_cover,c_prob,c_mess)
print(success,c_lsb)

c_extracted_msg = stc.extract(c_stego,c_lsb)
print(c_extracted_msg)
print(np.array(c_extracted_msg,dtype=int))
print(bits2str(list(c_extracted_msg)))

stego = np.reshape(np.array(c_stego),(n,n))

# plt.subplot(2,4,1)
# plt.imshow(cover, cmap='gray')
# plt.subplot(2,4,2)
# plt.imshow(stego, cmap='gray')
# plt.subplot(2,4,3)
# plt.imshow((stego-cover)/2, cmap='gray')
# plt.subplot(2,4,4)
# plt.imshow(prob_map, cmap='gray')
# plt.show()

