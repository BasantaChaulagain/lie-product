import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# Number of sample to be displayed in graph
start_index = 1
end_index =200

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
	add = np.add(np.array(a),np.array(b))
	return expMat(add)

def rhs(a,b,n):
	expA = expMat(1/float(n)*np.array(a))
	expB = expMat(1/float(n)*np.array(b))
	matmul = np.matmul(expA, expB)
	# raising the power of matrix to the number n.
	powered = np.power(matmul, n)
	return powered

def trace_distance(a,b):
	diff = np.subtract(a,b)
	result = (linalg.norm(diff,'nuc'))/2
	return result

def graph(n, td):
	f = plt.figure()
	plt.plot(n, td)
	plt.xlabel('n')
	plt.ylabel('Trace distance')
	plt.title('Trace distance vs n')
	# plt.show()
	f.savefig("graph.pdf")

def main():
	print ("Enter a square matrix A, in the form: [[a,b],[c,d]].")
	A = generate_matrix()
	print ("Enter a square matrix B, in the form: [[a,b],[c,d]].")
	B = generate_matrix()

	# A = [[1,-2],[1,0]]
	# B = [[0,2],[-1,0]]
	lhs_ = lhs(A,B)
	range_ = np.arange(start_index, end_index+1,1)
	td = []

	for n in range_:
		rhs_= rhs(A,B,n)
		td.append(trace_distance(lhs_, rhs_))
		print (n, td[n-start_index])
		
	graph(range_, td)


if __name__ == "__main__":
	main()