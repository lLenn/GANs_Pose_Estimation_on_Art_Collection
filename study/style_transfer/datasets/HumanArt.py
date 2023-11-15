import json
import itertools
import numpy as np
import copy
from collections import defaultdict
from torch.utils.data import Dataset
from utils import isArrayLike

# Split the set into keypoint and normal
class HumanArt(Dataset):
    def __init__(self, annotationFile=None):
        self.dataset = dict()
        self.annotations = dict()
        self.categories = dict()
        self.images = dict()
        self.imagesToAnnotations = defaultdict(list)
        self.categoriesToImages = defaultdict(list)
        
        print("loading annotations...")
        if not annotationFile == None:
            dataset = json.load(open(annotationFile))
            self.dataset = dataset
            self.createIndex()
        
    def createIndex(self):
        # create index
        print('creating indices...')
        if 'annotations' in self.dataset:
            for annotation in self.dataset['annotations']:
                self.imagesToAnnotations[annotation['image_id']].append(annotation)
                self.annotations[annotation['id']] = annotation

        if 'images' in self.dataset:
            for image in self.dataset['images']:
                self.images[image['id']] = image

        if 'categories' in self.dataset:
            for category in self.dataset['categories']:
                self.categories[category['id']] = category

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for annotation in self.dataset['annotations']:
                self.categoriesToImages[annotation['category_id']].append(annotation['image_id'])

    def getAnnotationIds(self, imageIds=[], categoryIds=[], isCrowd=None):
        imageIds = imageIds if isArrayLike(imageIds) else [imageIds]
        categoryIds = categoryIds if isArrayLike(categoryIds) else [categoryIds]

        if len(imageIds) == len(categoryIds) == 0:
            annotations = self.annotations
        else:
            if not len(imageIds) == 0:
                lists = [self.imagesToAnnotations[imageId] for imageId in imageIds if imageId in self.imagesToAnnotations]
                annotations = list(itertools.chain.from_iterable(lists))
            else:
                annotations = self.annotations
            annotations = annotations if len(categoryIds)  == 0 else [annotation for annotation in annotations if annotation['category_id'] in categoryIds]
        if not isCrowd == None:
            ids = [annotation['id'] for annotation in annotations if annotation['iscrowd'] == isCrowd]
        else:
            ids = [annotation['id'] for annotation in annotations]
        return ids
    
    def getCategoryIds(self, catNms=[], supNms=[], catIds=[]):
        catNms = catNms if isArrayLike(catNms) else [catNms]
        supNms = supNms if isArrayLike(supNms) else [supNms]
        catIds = catIds if isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset['categories']
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name']          in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id']            in catIds]
        ids = [cat['id'] for cat in cats]
        return ids
    
    def getImageIds(self, imageIds=[], categoryIds=[]):
        imageIds = imageIds if isArrayLike(imageIds) else [imageIds]
        categoryIds = categoryIds if isArrayLike(categoryIds) else [categoryIds]

        if len(imageIds) == len(categoryIds) == 0:
            ids = self.images.keys()
        else:
            ids = set(imageIds)
            for i, categoryId in enumerate(categoryIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.categoriesToImages[categoryId])
                else:
                    ids &= set(self.categoriesToImages[categoryId])
        return list(ids)
    
    def loadAnnotations(self, ids=[]):
        if isArrayLike(ids):
            return [self.annotations[id] for id in ids]
        elif type(ids) == int:
            return [self.annotations[ids]]
        
    def loadNumpyAnnotations(self, data):
        N = data.shape[0]
        annotations = []
        for i in range(N):
            if i % 1000000 == 0:
                print('{}/{}'.format(i,N))
            annotations += [{
                'image_id'  : int(data[i, 0]),
                'bbox'  : [ data[i, 1], data[i, 2], data[i, 3], data[i, 4] ],
                'score' : data[i, 5],
                'category_id': int(data[i, 6]),
                }]
        return annotations
        
    def loadImages(self, ids=[]):
        if isArrayLike(ids):
            return [self.images[id] for id in ids]
        elif type(ids) == int:
            return [self.images[ids]]
        
    def loadResults(self, resultFile):
        res = HumanArt()
        res.dataset['images'] = [img for img in self.dataset['images']]

        print('Loading and preparing results...')
        if type(resultFile) == str:
            annotations = json.load(open(resultFile))
        elif type(resultFile) == np.ndarray:
            annotations = self.loadNumpyAnnotations(resultFile)
        else:
            annotations = resultFile

        if 'bbox' in annotations[0] and not annotations[0]['bbox'] == []:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(annotations):
                bb = ann['bbox']
                x1, x2, y1, y2 = [bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]]
                ann['area'] = bb[2]*bb[3]
                ann['id'] = id+1
                ann['iscrowd'] = 0
        elif 'keypoints' in annotations[0]:
            res.dataset['categories'] = copy.deepcopy(self.dataset['categories'])
            for id, ann in enumerate(annotations):
                s = ann['keypoints']
                x = s[0::3]
                y = s[1::3]
                x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann['area'] = (x1-x0)*(y1-y0)
                ann['id'] = id + 1
                ann['bbox'] = [x0,y0,x1-x0,y1-y0]

        res.dataset['annotations'] = annotations
        res.createIndex()
        return res