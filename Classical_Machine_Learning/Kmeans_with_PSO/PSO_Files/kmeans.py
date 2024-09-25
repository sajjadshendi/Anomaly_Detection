import numpy

def calc_sse(centroids: numpy.ndarray, labels: numpy.ndarray, data: numpy.ndarray):
    distances = 0
    for i, c in enumerate(centroids):
        idx = numpy.where(labels == i)
        dist = numpy.sum((data[idx] - c)**2)
        distances += dist
    return distances
