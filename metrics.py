import numpy as np
import tensorflow as tf

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def binary_iou(y_true, y_pred):
    """Calculate the binary Intersection over Union (IoU), with handling for empty predictions."""
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    
    # Handle the case when both masks are empty (all zeros)
    if union == 0:
        return 1.0 if np.sum(y_true) == 0 and np.sum(y_pred) == 0 else 0.0
    
    return intersection / union

def dice_coef_test(y_true, y_pred):
    """Calculate the Dice coefficient, with handling for empty predictions."""
    intersection = np.sum(y_true * y_pred)
    denominator = np.sum(y_true) + np.sum(y_pred)
    
    # Handle the case when both masks are empty (all zeros)
    if denominator == 0:
        return 1.0 if np.sum(y_true) == 0 and np.sum(y_pred) == 0 else 0.0
    
    return 2 * intersection / denominator
