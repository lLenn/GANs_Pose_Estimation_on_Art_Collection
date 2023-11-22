def isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')