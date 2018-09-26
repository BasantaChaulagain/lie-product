import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Matrix generation
def generate_matrix():
	inp = input()
	if not check_square(inp):
		print ("Please enter a square matrix.")
	return np.matrix(inp)

#Check if the entered matrix is a square matrix.
def check_square(sq):
    rows = len(sq)
    for row in sq:
        if len(row) != rows:
            return False
    return True

# Calculation of matrix exponential
expMat = lambda x: linalg.expm(x)

def lhs(a,b):
	add = np.add(a,b)
	return expMat(add)

def rhs(a,b,n):
	expA = expMat(n*np.array(a))
	expB = expMat(n*np.array(b))
	matmul = np.matmul(expA, expB)
	# raising the power of matrix to the number n.
	powered = np.power(matmul, n)
	return powered

def graph(td, n):
	f = plt.figure()
	plt.plot(td, n, 'ro')
	plt.xlabel('n')
	plt.ylabel('Trace distance')
	plt.title('Trace distance vs n')
	f.savefig("graph.pdf")

def main():
	# print ("Enter a square matrix A, in the form: [[a,b],[c,d]].")
	# A = generate_matrix()
	# print ("Enter a square matrix B, in the form: [[a,b],[c,d]].")
	# B = generate_matrix()

	A = [[0,0],[1,0]]
	B = [[0,1],[0,0]]
	print (A)
	print (B)
	lhs(A,B)
	range_ = np.arange(0,20,1)
	rhs_list = []
	td = []

	for n in range_:
		rhs_list.append(rhs(A,B,n))

	print rhs_list
	
	for n in range_:
		td.append(trace_distance(lhs, rhs_list[n]))

	# td = [1,2,3,4]
	# n = [1,4,9,16]
	# graph(td, n)

if __name__ == "__main__":
	main()