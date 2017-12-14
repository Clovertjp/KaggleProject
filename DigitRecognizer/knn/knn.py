import csv
import heapq
import threading
import time
trainMap={};
with open('../train.csv') as csvfile:
	reader=csv.DictReader(csvfile);
	count=1;
	for row in reader:
		list=[];
		label=row["label"];
		for x in range(784):
			list.append(int(row["pixel"+str(x)]));
		trainMap[str(label)+"_"+str(count)]=list;
		count+=1;
		if count >= 1000:
			break;
valMap={};
print "load finish"

def cal(count,list):
	ticks = time.time()
	print count;
	tlist=[]
	for key,val in trainMap.items():
		sumData=0;
		tmap={};
		for i in range(len(val)):
			sumData+=pow((val[i]-list[i]),2);
		tmap["val"]=sumData;
		tmap["key"]=key;
		tlist.append(tmap);
	small=heapq.nsmallest(7,tlist,key=lambda s:s['val']);
	slist=[];
	for map in small:
		slist.append(map["key"][0]);
	sset=set(slist);
	num=100;
	v="";
	for item in sset:
		if(slist.count(item)<num):
			num=slist.count(item);
			v=item;
	subTime=time.time()-ticks;
	print str(count)+"------"+v+"------"+str(subTime);
	tmap={};
	tmap["val"]=v;
	tmap["key"]=count;
	mutex.acquire();
	fileList.append(tmap);
	mutex.release();
	if len(threads) >0:
		th=threads.pop();
		th.start();

fileList=[];
mutex = threading.Lock()	
def writeFile():
	i=1;
	f=open('f.txt','a');
	while i<=maxLine:
		
		if len(fileList)>0:
			mutex.acquire();
			va=fileList.pop();
			print va;
			mutex.release();
			f.write(str(va["key"])+"	"+va["val"]);
			f.write('\n');
			f.flush();
			i+=1;
		else:
			time.sleep(5);
	f.close();
	twrite.close();

threads = []
maxLine=1;
twrite = threading.Thread(target=writeFile,args=());
twrite.start();
with open('../test.csv') as testFile:
	reader=csv.DictReader(testFile);
	for row in reader:
		list=[];
		for x in range(784):
			list.append(int(row["pixel"+str(x)]));
		# t = threading.Thread(target=cal,args=(maxLine,list));
		# threads.append(t);
		cal(maxLine,list);
		maxLine+=1;

# for ti in range(15):
# 	thread=threads.pop();
# 	thread.start();

