import numpy as np

def feature_scale(data, avg, stdev):
	i=0
	j=0
	while(i<len(data[0])):
		j=0
		min = 100000000000
		max = -min
		sum = 0
		while(j<len(data)):

			if(data[j][i] < min ):
				min = data[j][i]

			if(data[j][i] > max ):
				max = data[j][i]

			sum = sum + data[j][i]
			j += 1

		avg[i] = sum/len(data)
		stdev[i] = max- min
		i+=1



def scale_data(data, avg, stdev):
	i=0
	j=0
	while(i<len(data[0])):
		j=0
		while(j<len(data)):
			data[j][i] = (data[j][i] - avg[i])/stdev[i]
			j+=1
		i+=1

def hypothesis(arr, theta):
	sum = theta[0]
	i=0
	while(i<len(arr)-1):
		sum = sum + arr[i]*theta[i+1]
		i+=1
	return sum


def cost_function(data, theta):
	i=0
	sum = 0
	while(i<len(data)):
		h = hypothesis(data[i], theta)
		sum = sum + (h-data[i][len(data[i]) - 1])**2
		i+=1
	result = sum/(2*len(data))
	return result

def grad_descent(data, theta, alpha):
	i=0
	temp = theta
	while(i<len(theta)):
		j=0
		sum = 0
		
		while(j<len(data)):
			
			h = hypothesis(data[j], theta)
			if(i==0):
				x = 1
			else:
				x = data[j][i-1]


			sum = sum + (h - data[j][ len(data[j]) - 1 ])*x
			j+=1

		temp[i] = temp[i] -((alpha*sum)/len(data))
		i+=1
	theta = temp

location = input('Enter the location of training set:   ')
work_set = np.genfromtxt(location, delimiter = ',', dtype =np.double)
col = len(work_set[0])

avg = np.empty((col), dtype =np.double)  #avg value
stdev = np.empty((col), dtype =np.double)  #standard deviation

#feature_scale(work_set, avg, stdev)

scale_data(work_set, avg, stdev)
theta = np.full((col), 0, dtype =np.double)

alpha = 1.5
pres_cost = 10000
itr = 0

while(itr<100):
	grad_descent(work_set, theta, alpha)
	itr+=1

print("Final hypothesis is :: \n",theta[0],end=' ')
i=1
while(i<len(theta)):
	print('+ ',theta[i],'x',i, end = ' ')
	i+=1

location = input('\nEnter the location of testing set:   ')
work_set = np.genfromtxt(location, delimiter = ',', dtype =np.double)

feature_scale(work_set, avg, stdev)
scale_data(work_set, avg, stdev)

nf = len(work_set[0])-1

while(i<len(work_set)):
	h = hypothesis(work_set[i],theta)
	h = h*stdev[nf] + avg[nf]
	print(h)
	i+=1
