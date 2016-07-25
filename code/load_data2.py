from PIL import Image
import random
from glob import glob
import numpy as np
import theano
import gzip, cPickle
from itertools import chain
DEFAULT=100 


def find_max_dim(wordsdir):
    def img_size_helper(maxv,im):
        img=Image.open(im) 
        w,h = img.size
        img.close()
        curr_max =  w if w > h else h
        return maxv if  maxv[0] > curr_max else (curr_max,im)
    max_dim = 0,'NIL'
    for im in wordsdir:
        path = im + '/*.png'
        path = glob(path)
        max_dim_in_a =  reduce(img_size_helper,path,max_dim)
        max_dim = max_dim if max_dim[0] > max_dim_in_a[0] else max_dim_in_a
    return max_dim
def find_avg_dim(wordsdir):
    def img_avg_helper(a,im):
        img = Image.open(im)
        w,h = img.size
        img.close()
        return a+w+h
        
    sum = 0
    size = 0
    for im in wordsdir:
        path = im+'/*.png'
        path=glob(path)
        size+= 2*len(path)
        sum += reduce(img_avg_helper,path,0)
    print sum , size
    return sum/size
def process(wordsdir,resize=100):
    def process_img(im,sz):
        img = Image.open(im).convert('L') #convert to greyscale
        normalized_img = img.resize((sz,sz),Image.ANTIALIAS)        
        imgarr = np.asarray(normalized_img)
        img.close()
        return imgarr
    x_set,y_set =[], []
    for a,b in wordsdir:
        path = glob(a+'/*.png')
        image_list = [process_img(i,resize) for i in path] 
        x_set+=(image_list)
        y_set+=(map(lambda x: b, image_list))
    return x_set,y_set

def load_data(dataset=u'/media/allCombined/train/*',size=DEFAULT):
    words = glob(dataset)
    word_labels = zip(words,(range(len(words)))) 
    return process(word_labels,size)

def find_median_dim(setA,setB):
   set_a_dir = glob(setA)
   set_b_dir = glob(setB)
   words = set_a_dir+set_b_dir
   dims=[]
   def process(im):
       img=Image.open(im)
       w,h = img.size
       img.close()
       return w if w > h else h
   for a in words:
       path=glob(a+'/*.png')
       dims_of_a = [process(i) for i in path]
       dims+= dims_of_a
   return int(np.median(np.asarray(dims)))
       
#create-data() -> creates a pickle with our data [(Tr_x,Tr_y),(Te_x,Te_y)] 
def create_data():
    print 'in create_data'
    test_dir, train_dir = create_train_test();
    test_dir = u'allCombined/test/*'
    train_dir = u'allCombined/train/*'
    #n_dim = find_n_dim(train_dir,test_dir)
    median_dim=DEFAULT
    train_set = load_data(train_dir,median_dim)
    test_set = load_data(test_dir,median_dim)
    dataset = [train_set, test_set]
    f = gzip.open('arabic_pictures.pkl.gz','wb')
    cPickle.dump(dataset,f,protocol=2)
    f.close() 

def get_data(dataset='arabic_pictures.pkl.gz'):
    #todo: check if the pickle exists, if so just unzip it and return the stuff
    create_data()
    f=gzip.open(dataset,'rb')
    data = cPickle.load(f)
    f.close()
    return data

