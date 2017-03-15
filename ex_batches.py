# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 21:34:21 2017

@author: diz
"""

import math
def batches(batch_size, features, labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param labels: List of labels
    :return: Batches of (Features, Labels)
    """
    sz = len(features)
    assert sz == len(labels)
    batches = []
    for start in range(0, sz, batch_size):
        batches.append([features[start:start+batch_size],
                        labels[start:start+batch_size]])
        
    return batches
    