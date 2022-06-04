from keras.layers import *
import tensorflow as tf
import model23


def CoT(x, filters, reduction):
    Key_map = Conv3D(filters, 3, padding='same')(x)
    Query = x
    Value_map = Conv3D(filters, 1, padding='same')(x)
    Y = Concatenate()([Key_map, Query])
    Y = Conv3D(filters * 2, 1, padding='same')(Y)
    Y = Conv3D(filters, 1, padding='same')(Y)
    Y = Multiply()([Y, Value_map])

    Y = model23.P.SE(Y, r=reduction)
    return Y


def identity_block(input_tensor, kernel_size, filters, stage, block, reduction,
                   use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(nb_filter1, 1, name=conv_name_base + '2a',
               kernel_initializer=tf.keras.initializers.he_normal)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = LeakyReLU(0.1)(x)

    # x = Conv3D(nb_filter2, kernel_size, padding='same',
    #            name=conv_name_base + '2b', use_bias=use_bias,
    #            kernel_initializer=keras.initializers.he_normal)(x)
    # x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    # x = LeakyReLU(0.1_1)(x)
    x = CoT(x, nb_filter2, reduction)

    x = Conv3D(nb_filter3, 1, name=conv_name_base + '2c',
               kernel_initializer=tf.keras.initializers.he_normal)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)
    # x = model23.P.SE(x, r=reduction, high=high)

    x = Add()([x, input_tensor])
    x = LeakyReLU(0.1, name='res' + str(stage) + block + '_out')(x)
    # x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, reduction,
               strides=2, use_bias=True, train_bn=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv3D(nb_filter1, 1, strides=strides,
               name=conv_name_base + '2a', use_bias=use_bias,
               kernel_initializer=tf.keras.initializers.he_normal)(input_tensor)
    x = BatchNormalization(name=bn_name_base + '2a')(x, training=train_bn)
    x = LeakyReLU(0.1)(x)

    # x = Conv3D(nb_filter2, kernel_size, padding='same',
    #            name=conv_name_base + '2b', use_bias=use_bias,
    #            kernel_initializer=keras.initializers.he_normal)(x)
    # x = BatchNormalization(name=bn_name_base + '2b')(x, training=train_bn)
    # x = LeakyReLU(0.1_1)(x)
    x = CoT(x, nb_filter2, reduction)

    x = Conv3D(nb_filter3, 1, name=conv_name_base + '2c', use_bias=use_bias,
               kernel_initializer=tf.keras.initializers.he_normal)(x)
    x = BatchNormalization(name=bn_name_base + '2c')(x, training=train_bn)
    # x = model23.P.SE(x, r=reduction, high=high)

    shortcut = Conv3D(nb_filter3, 1, strides=strides,
                      name=conv_name_base + '1_1', use_bias=use_bias,
                      kernel_initializer=tf.keras.initializers.he_normal)(input_tensor)
    shortcut = BatchNormalization(name=bn_name_base + '1_1')(shortcut, training=train_bn)

    x = Add()([x, shortcut])
    x = LeakyReLU(0.1, name='res' + str(stage) + block + '_out')(x)
    # x = Activation('relu', name='res' + str(stage) + block + '_out')(x)
    return x


def get_resnet_SE(input_image, filters, name, reduction, stage5=False, train_bn=True, block_count=5):
    x = ZeroPadding3D(1)(input_image)
    x = Conv3D(filters, 3, strides=2, name=name + 'conv1', use_bias=True,
               kernel_initializer=tf.keras.initializers.he_normal)(x)
    x = BatchNormalization(name=name + 'bn_conv1')(x, training=train_bn)
    C1 = x = LeakyReLU(0.1)(x)
    # Height/4,Width/4,64
    x = MaxPooling3D(3, strides=2, padding="same")(x)
    # Stage 2
    x = conv_block(x, 3, [filters, filters, filters * 2], stage=2, block=name + 'a',
                   strides=1, reduction=reduction)
    x = identity_block(x, 3, [filters, filters, filters * 2], stage=2,
                       block=name + 'b', train_bn=train_bn, reduction=reduction)
    # Height/4,Width/4,256
    C2 = x = identity_block(x, 3, [filters, filters, filters * 2], stage=2,
                            block=name + 'c', train_bn=train_bn, reduction=reduction)
    # Stage 3
    x = conv_block(x, 3, [filters * 2, filters * 2, filters * 4], stage=3,
                   block=name + 'a', train_bn=train_bn, reduction=reduction)
    x = identity_block(x, 3, [filters * 2, filters * 2, filters * 4], stage=3,
                       block=name + 'b', train_bn=train_bn, reduction=reduction)
    x = identity_block(x, 3, [filters * 2, filters * 2, filters * 4], stage=3,
                       block=name + 'c', train_bn=train_bn, reduction=reduction)
    # Height/8,Width/8,512
    C3 = x = identity_block(x, 3, [filters * 2, filters * 2, filters * 4], stage=3,
                            block=name + 'd', train_bn=train_bn, reduction=reduction)
    # Stage 4
    x = conv_block(x, 3, [filters * 4, filters * 4, filters * 8], stage=4,
                   block=name + 'a', train_bn=train_bn, reduction=reduction)
    # block_count = 12
    for i in range(block_count):
        x = identity_block(x, 3, [filters * 4, filters * 4, filters * 8], stage=4,
                           block=name + chr(98 + i), train_bn=train_bn, reduction=reduction)
    # Height/16,Width/16,1024
    C4 = x
    # Stage 5
    if stage5:
        x = conv_block(x, 3, [filters * 8, filters * 8, filters * 16], stage=5,
                       block=name + 'a', train_bn=train_bn, reduction=reduction)
        x = identity_block(x, 3, [filters * 8, filters * 8, filters * 16], stage=5,
                           block=name + 'b', train_bn=train_bn, reduction=reduction)
        # Height/32,Width/32,2048
        C5 = x = identity_block(x, 3, [filters * 8, filters * 8, filters * 16],
                                stage=5, block=name + 'c', train_bn=train_bn, reduction=reduction)
    else:
        C5 = None
    return [C1, C2, C3, C4, C5]


if __name__ == '__main__':
    inputs = Input((64, 256, 256, 1))
    high = Input(64)
    out = get_resnet_SE(inputs, 8, 'CT', reduction=4, stage5=True)
    print(out)