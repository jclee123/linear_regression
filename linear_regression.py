#sample inputs (y is output array, x can be any n+1 dimensional array (first array set to all 1))
y=[2,2,6,9,11,12,14]
x=[[1,1,1,1,1,1,1],[1,2,3,4,5,7,8],[1,3,5,8,9,12,13]]


#x_0*theta_0 + x_1*theta_1+x_2*theta_2... and so on consists of the hypothesis
#x_0 is always assumed to be 1.
theta=[]
error=[]
#hypothesis
hyp=0

m=len(x)
#constant
alpha=0.01/len(y)

for i in range(m):
    theta.append(0)
    error.append(0)

#repeat until convergence
for i in range(10000):
    #loop through all training examples to compute gradient step
    for j in range(len(y)):
        #calculate errors for all theta values, and update them simulateously
        for k in range(m):
            #sum hypothesis depending on number of parameters
            for l in range(m):
                hyp+=theta[l]*x[l][j]
            #error total from all hypothesis
            error[k]+=(hyp-y[j])*x[k][j]
            hyp=0
    #gradient step
    for p in range(m):
        theta[p]-=alpha*error[p]
        error[p]=0

#outputs
for output in range(m):
    print("x_"+str(output)+"*"+str(theta[output]))

#for output in range(len(y)):
#    print(theta[0]+theta[1]*x[1][output]+theta[2]*x[2][output])
