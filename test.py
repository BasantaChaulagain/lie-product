# This is a file created for testing. It has nothing to do with
# the main code. Please ignore this.


from scipy import linalg
import numpy as np


expMat = lambda x: linalg.expm(x)

# A = [[1,2],[1,0]]
B = [[-2,2],[-1,0]]

# for n in range(1,10):
# 	nor = 1/float(n)*np.array(A)
# 	# print linalg.norm(nor,'nuc')
# 	# print np.power(nor,n)
# 	# print linalg.norm(np.power(nor,n),'nuc')
# 	e = expMat(nor)

# 	print linalg.norm(e,'nuc')

A = np.array(input())
print (A)
print (type(A))
print (np.matrix(str(A)))