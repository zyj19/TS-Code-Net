import tensorflow as tf
import keras.backend as K
from tensorflow.python.ops import array_ops
import numpy as np


def GDL_add_focal(gamma=2., alpha=.25):

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1 + K.epsilon())) - K.mean(
            (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))

    def generalized_dice_coeff(y_true, y_pred):
        # Ncl = y_pred.shape[-1]
        # w = K.zeros(shape=(Ncl,))
        w = K.sum(y_true, axis=(0, 1, 2, 3))
        w = 1 / (w ** 2 + 0.000001)
        # Compute gen dice coef:
        numerator = y_true * y_pred
        numerator = w * K.sum(numerator, (0, 1, 2, 3, 4))
        numerator = K.sum(numerator)
        denominator = y_true + y_pred
        denominator = w * K.sum(denominator, (0, 1, 2, 3, 4))
        denominator = K.sum(denominator)
        gen_dice_coef = 2 * numerator / denominator
        return gen_dice_coef

    def generalized_dice_loss(y_true, y_pred):
        return 1 - generalized_dice_coeff(y_true, y_pred) + 10 * focal_loss_fixed(y_true, y_pred)

    return generalized_dice_loss
