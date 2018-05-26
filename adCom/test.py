import random
b=[]
for a in open('test1.csv','r').read().split('\n')[1:-1]:
	score =random.random()
	b.append(a+','+str(score)[:5])

with open('sub.csv','w') as f:
	f.write('\n'.join(b))