import math
import numpy as np
from random import random

__all__ = ["process_in_batch", "crop", "random_flip", "random_scaling_2d",
           "random_rotation_2d", "random_intensity_remap"]

def process_in_batch(processes, *arg):

    if not isinstance(processes, (list,tuple)):
        processes = [processes]

    aux = []
    for sample in zip(*arg):
        for process in processes:
            sample = process(sample)
        aux.append(sample)

    aux = tuple(x for x in zip(*aux))
    return aux if len(arg)>1 else aux[0]

def get_shape(images):
    if isinstance(images, (list,tuple)):
        image_shape = images[0].shape
    else:
        image_shape = images.shape
    return image_shape

def _crop(image_shape, shape=(512,512), method='random'):

    for image_dim, dim in zip(image_shape, shape):
        assert image_dim > dim, "image is smaller than the cropped shape!"

    if method=='random':
        f = lambda image_dim, crop_dim: np.random.randint(0, image_dim-crop_dim + 1)
    elif method=='center':
        f = lambda image_dim, crop_dim: (image_dim - crop_dim)//2
    elif method=='upperleft':
        f = lambda image_dim, crop_dim: 0
    else:
        raise ValueError("unknown crop method {}".format(method))

    inits = [f(image_shape[i], shape[i]) for i in range(len(shape))]
    tuple_slice = tuple(slice(init,init + dim) for init,dim in zip(inits, shape))
    return tuple_slice

def crop(images, shape=(512,512), method='random'):
    """ Crop image(s)

    If param images is an iterable (a.k.a list, tuple, ..)
    the method applys the exact same transformation to all
    elements in the iterable.

    Parameters
    ----------
    images : numpy.ndarray or iterable of numpy.ndarray (N,D1,D2,..,Channels)
        image or list of images.
    shape : tuple or list
        crop size
    method : str
        'random' or 'center' or 'upperleft'

    Return
    ------
    numpy.ndarray or iterable of numpy.ndarray
    """
    image_shape = get_shape(images)

    tuple_slice = _crop(image_shape, shape, method)
    if isinstance(images, (list,tuple)):
        return [image[tuple_slice] for image in images] + [tuple_slice]

    return images[tuple_slice]

def _random_flip(axis, p):

    operations = []
    for axe,p in zip(axis,p):
        if random()<p:
            operations.append(lambda x: np.flip(x, axis=axe))

    def f(x):
        _x = x.copy()
        for op in operations:
            _x = op(_x)
        return _x

    return f

def random_flip(images, axis=None, p=None):
    """ Random flip image(s) along axis

    If param images is an iterable (a.k.a list, tuple, ..)
    the method applys the exact same transformation to all
    elements in the iterable.

    Parameters
    ----------
    images : numpy.ndarray or iterable of numpy.ndarray (N,D1,D2,..,Channels)
        image or list of images.
    axis : int or iterable (N,)
        index of the axis to flip with probability p
    p : float or iterable (N,)
        flipping probability

    Return
    ------
    numpy.ndarray or iterable of numpy.ndarray
    """

    image_shape = get_shape(images)

    if axis is None:
        axis = list(range(len(image_shape)))
    if p is None:
        p = list(0.5 for _ in axis)

    process = _random_flip(axis, p)

    if isinstance(images, (list,tuple)):
        return [process(image) for image in images]

    return process(images)

def _random_scaling_2d(scales=(1,2,3,4), p=(0.25,0.25,0.25,0.25)):
    import cv2

    s = np.random.choice(scales, p=p)
    f = lambda x: cv2.resize(x, dsize=None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)

    return f

def random_scaling_2d(images, scales=(1,2,3,4), p=(0.25,0.25,0.25,0.25)):
    """ Random scaling image(s)

    If param images is an iterable (a.k.a list, tuple, ..)
    the method applys the exact same transformation to all
    elements in the iterable.

    Parameters
    ----------
    images : numpy.ndarray or iterable of numpy.ndarray (N,D1,D2,..,Channels)
        image or list of images.
    scales : iterable (M,)
        possible scales
    p : iterable (M,)
        probability of occurence for each scale

    Return
    ------
    numpy.ndarray or iterable of numpy.ndarray
    """

    process = _random_scaling_2d(scales, p)

    if isinstance(images, (list,tuple)):
        return [process(image) for image in images]

    return process(images)

def _random_rotation_2d(range=(-180,180)):
    from scipy import ndimage

    theta = np.random.randint(*range)
    rot_image = lambda image: ndimage.rotate(image, theta, axes=(0,1), reshape=False, order=3)
    rot_label = lambda label: ndimage.rotate(label, theta, axes=(0,1), reshape=False, order=0,
                                             mode='constant', cval=255)
    return rot_image, rot_label

def random_rotation_2d(images, range=(-180,180)):
    """ Random scaling image(s)

    !!! Special case
    If images is  list/tuple, the second element of the list is
    rotate using nearest neighbour interpolation, the rest
    with interpolation of order 3.
    As an example: images=[image,label,weight,image2,image3,..]
    !!!

    Parameters
    ----------
    images : numpy.ndarray or iterable of numpy.ndarray (N,D1,D2,..,Channels)
        image or list of images.
    scales : iterable (M,)
        possible scales
    p : iterable (M,)
        probability of occurence for each scale

    Return
    ------
    numpy.ndarray or iterable of numpy.ndarray
    """

    rot_image, rot_label = _random_rotation_2d(range)

    if isinstance(images, (list,tuple)):
        if len(images)==2:
            return [rot_image(images[0]), rot_label(images[1])]
        elif len(images)>2:
            return [rot_image(images[0]), rot_label(images[1])] + [rot_image(i) for i in images[2:]]

    return rot_image(images)

def _random_intensity_remap(image, max_z):

    # based on: full-resolution residual networks, appendix A
    z_over_sqrt2 = (random()-0.5)*max_z/math.sqrt(2)
    gamma = math.log(z_over_sqrt2+0.5)/math.log(-z_over_sqrt2+0.5)
    image = np.power(image, gamma)
    return image

def random_intensity_remap(images, max_z):
    """ Random intensity remap

    Only applys for images.

    Parameters
    ----------
    images : numpy.ndarray or iterable of numpy.ndarray (N,D1,D2,..,Channels)
        image or list of images.
    max_z : float

    Return
    ------
    numpy.ndarray or iterable of numpy.ndarray
    """
    image = images[0]
    for channel in range(image.shape[-1]):
        image[...,channel] = _random_intensity_remap(image[...,channel], max_z)

    return [image]
