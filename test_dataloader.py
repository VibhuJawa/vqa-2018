from dataloaders.VQADataset import VQADataset
import time
data_dir = '/home-3/pmahaja2@jhu.edu/scratch/vqa2018_data'
opt = {'dir':data_dir,'images':'Images','nans':2000, 'sampleans':True, 'maxlength':26, 'minwcount':0, 'nlp':'mcb','pad':'left','trainsplit':'train'}

begin = time.time()
D = VQADataset("train", opt)
dl = D.data_loader(batch_size=32, num_workers=4,shuffle=False)
print("Time to load Data loader : ",time.time() - begin)
for i in dl:
    pass
print("Time for iterating over {} questions : {} \nTime per question : {}".format(len(D), time.time() - begin, len(D)/(time.time() - begin)))

#begin = time.time()
#D = VQADataset("test", opt)
#dl = D.data_loader(batch_size=3, num_workers=4,shuffle=False)
#print("Time to load Data loader : ",time.time() - begin)
#for i in dl:
#    break
#print("Time for iterating over {} questions : {} \nTime per question : {}".format(len(D), time.time() - begin, len(D)/(time.time() - begin)))
#
#begin = time.time()
#D = VQADataset("dev", opt)
#dl = D.data_loader(batch_size=3, num_workers=4,shuffle=False)
#print("Time to load Data loader : ",time.time() - begin)
#for i in dl:
#    break
#print("Time for iterating over {} questions : {} \nTime per question : {}".format(len(D), time.time() - begin, len(D)/(time.time() - begin)))
#
#begin = time.time()
#D = VQADataset("testdev", opt)
#dl = D.data_loader(batch_size=3, num_workers=4,shuffle=False)
#print("Time to load Data loader : ",time.time() - begin)
#for i in dl:
#    break
#print("Time for iterating over {} questions : {} \nTime per question : {}".format(len(D), time.time() - begin, len(D)/(time.time() - begin)))
