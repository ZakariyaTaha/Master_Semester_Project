import numpy as np
from collections import namedtuple
ExtDataPoint = namedtuple("ExtDataPoint", ['image', 'label', 'graph'])

def cropDataset(dataset, crop_size):
    image_size = dataset[0].image.shape
    r = np.prod(image_size / crop_size)
    new_dataset = []
    for _ in range(3*int(r)):
        images = [dp.image for dp in dataset]
        labels = [dp.label for dp in dataset]

        # --- augmentation ---
        f = []
        f.append( lambda sample: nt.crop(sample, crop_size, "random") )
        f.append( lambda sample: nt.random_flip(sample, axis=(0,1,2), p=(0.5,0.5,0.5)) )
        images, labels = nt.process_in_batch(f, images, labels)
        for im, lbl in zip(images, labels):
            g = graph_from_skeleton(lbl)
            new_dataset.append(ExtDataPoint(im, lbl, g))
    
    return new_dataset
if __name__ == "__main__":
    continue