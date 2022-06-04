import keras.layers as KL
from keras.layers import *
import keras.backend as K
from keras.models import Model
import ResNet101_SE
import tensorflow as tf


def fusion(feature_map, x, filters, name):
    y = KL.Concatenate(name=name + 'Con_fusion')([feature_map, x])
    y = KL.Conv3D(filters, 3, strides=1, padding='same', name=name + 'Conv_fusion',
                  kernel_initializer=tf.keras.initializers.he_normal)(y)
    y = KL.BatchNormalization(name=name + 'BN_fusion')(y, training=True)
    y = KL.LeakyReLU(0.1, name=name + 'L_fusion')(y)

    size = feature_map.shape[2]
    x = KL.Concatenate(axis=1, name=name + 'Con_fusion_0')([KL.Reshape((1, size, size, -1))(feature_map[:, 0, :, :, :]),
                                                            KL.Reshape((1, size, size, -1))(x[:, 0, :, :, :])])
    for i in range(1, feature_map.shape[1]):
        t = KL.Concatenate(axis=1, name=name + 'Con_1_fusion_' + str(i))([KL.Reshape((1, size, size, -1))(feature_map[:,
                                                                                                          i, :, :, :]),
                                                                          KL.Reshape((1, size, size, -1))(
                                                                              x[:, i, :, :, :])])
        x = KL.Concatenate(axis=1, name=name + 'Con_fusion_' + str(i))([x, t])

    x = KL.ZeroPadding3D((0, 1, 1), name=name + 'Zero')(x)
    x = KL.Conv3D(filters, (2, 3, 3), strides=(2, 1, 1), name=name + 'Conv1_fusion',
                  kernel_initializer=tf.keras.initializers.he_normal)(x)
    x = KL.BatchNormalization(name=name + 'BN1_fusion')(x, training=True)
    x = KL.LeakyReLU(0.1, name=name + 'L1_fusion')(x)
    x = KL.Multiply(name=name + 'Mu_fusion')([y, x])
    # x = KL.Activation('relu')(x)
    return x


class P:
    @staticmethod
    def backbone_SE(inputs, name, filters, reduction, block_count=5):
        out = ResNet101_SE.get_resnet_SE(inputs, filters, name, reduction, stage5=True, train_bn=True,
                                         block_count=block_count)
        return out

    @staticmethod
    def Block(x, kernel, filters, name, is_dilated=False, rate=(1, 1, 1)):
        if is_dilated:
            dilated_rated = rate
        else:
            dilated_rated = (1, 1, 1)

        x = KL.Conv3D(filters, kernel, padding='same', dilation_rate=dilated_rated,
                      kernel_initializer=tf.keras.initializers.he_normal, name=name + 'Conv')(x)
        x = KL.BatchNormalization(name=name + 'Ba')(x, training=True)
        x = KL.LeakyReLU(0.1, name=name + 'Le')(x)
        return x

    @staticmethod
    def RFs(x, filters, name, reduction):
        # filters = filters / 4
        x0_1 = x0 = P.Block(x, 1, filters, name=name + '1_1')
        x0 = P.Block(x0, (3, 1, 3), filters, name=name + '2')
        x0 = P.Block(x0, (3, 3, 1), filters, name=name + '3')
        x0 = P.Block(x0, 3, filters, is_dilated=True, rate=(3, 3, 3), name=name + '6')
        x0 = P.SE(x0, r=reduction)
        x0 = Add()([x0, x0_1])

        x1_1 = x1 = P.Block(x, 1, filters, name=name + '7')
        x1 = P.Block(x1, (5, 1, 5), filters, name=name + '8')
        x1 = P.Block(x1, (5, 5, 1), filters, name=name + '9')
        x1 = P.Block(x1, 5, filters, is_dilated=True, rate=(5, 5, 5), name=name + '12')
        x1 = P.SE(x1, r=reduction)
        x1 = Add()([x1, x1_1])

        x2_1 = x2 = P.Block(x, 1, filters, name=name + '13')
        x2 = P.Block(x2, (7, 1, 7), filters, name=name + '14')
        x2 = P.Block(x2, (7, 7, 1), filters, name=name + '15')
        x2 = P.Block(x2, 7, filters, is_dilated=True, rate=(7, 7, 7), name=name + '18')
        x2 = P.SE(x2, r=reduction)
        x2 = Add()([x2, x2_1])

        x3 = P.Block(x, 1, filters, name=name + '19')
        # filters *= 4

        x = KL.Concatenate(name=name + 'Conc')([x0, x1, x2, x3])
        x = P.CE_block(x, name)
        x = KL.Dropout(0.5, name=name + 'Drop')(x)
        x = P.Block(x, 1, filters, name=name + '20')

        x4 = P.Block(x, 1, filters, name=name + '21')
        x = KL.Add(name=name + 'add')([x, x4])
        x = KL.LeakyReLU(0.1, name=name + 'Le')(x)
        return x

    @staticmethod
    def PDC(x0, x1, x2, x3, filters, name, size=(2, 2, 2)):
        x3 = KL.UpSampling3D(size, name=name + 'Up')(x3)
        x3_1 = P.Block(x3, 3, filters, name=name + '1_1')
        x3_1 = KL.Multiply(name=name + 'Mu')([x3_1, x2])
        x3_2 = P.Block(x3, 3, filters, name=name + '2')
        x3_1 = KL.Concatenate(name=name + 'Conc')([x3_1, x3_2])
        # x3_1 = KL.Dropout(0.5)(x3_1)
        x3_1 = P.Block(x3_1, 3, filters, name=name + '3')
        x3_1 = KL.UpSampling3D(2, name=name + 'Up1')(x3_1)
        x3_1 = P.Block(x3_1, 3, filters, name=name + '4')
        x3_3 = KL.UpSampling3D(2, name=name + 'Up2')(x3)
        # x3_3 = KL.UpSampling3D(2)(x3_3)
        x3_3 = P.Block(x3_3, 3, filters, name=name + '5')
        x2 = KL.UpSampling3D(2, name=name + 'Up3')(x2)
        x2 = P.Block(x2, 3, filters, name=name + '6')

        if x1 is not None:
            x2 = KL.Multiply(name=name + 'Mu1')([x3_3, x2, x1])
        else:
            x2 = KL.Multiply(name=name + 'Mu2')([x3_3, x2])
        if x0 is not None:
            x0 = KL.Concatenate(name=name + 'Conc1')([x3_1, x2, x0])
        else:
            x0 = KL.Concatenate(name=name + 'Conc2')([x3_1, x2])
        x0 = KL.Dropout(0.5, name=name + 'Drop')(x0)
        x0 = P.Block(x0, 3, filters, name=name + '7')
        x0 = P.Block(x0, 3, filters, name=name + '8')
        x0 = KL.Conv3D(filters, 1, padding='same',
                       kernel_initializer=tf.keras.initializers.he_normal,
                       name=name + '_Conv')(x0)
        x0 = KL.BatchNormalization(name=name + '_Ba')(x0, training=True)
        x0 = KL.LeakyReLU(0.1, name=name + '_Le')(x0)
        return x0

    @staticmethod
    def resize(image, x):
        size = image.shape[1:]
        image = tf.transpose(image, (0, 2, 3, 1))
        image = tf.image.resize(image, (int(size[1] / x), int(size[2] / x)))
        image = tf.transpose(image, (0, 3, 1, 2))
        image = tf.image.resize(image, (int(size[0] / x), int(size[1] / x)))
        return image

    @staticmethod
    def PDC_1(x0, x1, x2, x3, filters, name, Seg, size=(2, 2, 2)):
        # x3 = KL.UpSampling3D(size, name=name + 'Up')(x3)

        x3_1 = P.Block(x3, 3, filters, name=name + '1_1')
        Seg1 = Seg
        x = Seg.shape[1] / (x2.shape[1] / 2)
        Seg1 = P.resize(Seg1[..., 0], x)[..., None]
        x3_1 = P.FM(x3_1, Seg1, x2, name=name + 'FFM')
        # x3_1 = KL.UpSampling3D(size, name=name + 'Up')(x3_1)

        x3_2 = P.Block(KL.UpSampling3D(size, name=name + 'Up')(x3), 3, filters, name=name + '2')
        x3_1 = KL.Concatenate(name=name + 'Conc')([x3_1, x3_2])
        # x3_1 = KL.Dropout(0.5)(x3_1)
        x3_1 = P.Block(x3_1, 3, filters, name=name + '3')
        x3_1 = KL.UpSampling3D(2, name=name + 'Up1')(x3_1)
        x3_1 = P.Block(x3_1, 3, filters, name=name + '4')
        x3_3 = KL.UpSampling3D(2, name=name + 'Up2')(x3)
        # x3_3 = KL.UpSampling3D(2)(x3_3)
        x3_3 = P.Block(x3_3, 3, filters, name=name + '5')
        # x2 = KL.UpSampling3D(2, name=name + 'Up3')(x2)
        # x2 = P.Block(x2, 3, filters, name=name + '6')

        if x1 is not None:
            x = Seg.shape[1] / (x1.shape[1] / 2)
            Seg2 = P.resize(Seg[..., 0], x)[..., None]
            x2 = P.FM(x2, Seg2, x1, name=name + 'FFM')
            x2 = P.FM(x3_3, Seg2, x2, name=name + 'FFM')
        else:
            x2 = KL.Multiply(name=name + 'Mu2')([x3_3, x2])
        if x0 is not None:
            x0 = KL.Concatenate(name=name + 'Conc1')([x3_1, x2, x0])
        else:
            x2 = KL.UpSampling3D(2, name=name + 'Up3')(x2)
            x2 = P.Block(x2, 3, filters, name=name + '6')
            x0 = KL.Concatenate(name=name + 'Conc2')([x3_1, x2])
        x0 = KL.Dropout(0.5, name=name + 'Drop')(x0)
        x0 = P.Block(x0, 3, filters, name=name + '7')
        x0 = P.Block(x0, 3, filters, name=name + '8')
        x0 = KL.Conv3D(filters, 1, padding='same',
                       kernel_initializer=tf.keras.initializers.he_normal,
                       name=name + '_Conv')(x0)
        x0 = KL.BatchNormalization(name=name + '_Ba')(x0, training=True)
        x0 = KL.LeakyReLU(0.1, name=name + '_Le')(x0)
        return x0

    @staticmethod
    def Up(x, name):
        filters = x.shape[-1] / 2
        x = KL.UpSampling3D(name=name + 'Up')(x)
        x = P.Block(x, 3, filters, name=name + '1_1')
        return x

    @staticmethod
    def Fusion(out0, out1, out2, out3, out4,
               out0_1, out1_1, out2_1, out3_1, out4_1):
        out0 = fusion(out0, out0_1, out0.shape[-1], '0')
        out1 = fusion(out1, out1_1, out1.shape[-1], '1_1')
        out2 = fusion(out2, out2_1, out2.shape[-1], '2')
        out3 = fusion(out3, out3_1, out3.shape[-1], '3')
        out4 = fusion(out4, out4_1, out4.shape[-1], '4')
        return out0, out1, out2, out3, out4

    @staticmethod
    def build():
        filters = 16
        reduction = 8
        inputs = KL.Input((64, 256, 256, 1))
        inputs_1 = KL.Input((64, 256, 256, 1))
        # high = Input(64)
        out0, out1, out2, out3, out4 = P.backbone_SE(inputs, 'CT', filters, reduction, block_count=5)
        out0_1, out1_1, out2_1, out3_1, out4_1 = P.backbone_SE(inputs_1, 'PET',
                                                               filters, reduction, block_count=5)

        out0, out1, out2, out3, out4 = P.Fusion(out0, out1, out2, out3, out4,
                                                out0_1, out1_1, out2_1, out3_1, out4_1)

        # print(out0.shape)
        filter_4 = 32
        # 2, 8, 8, 128
        X4 = P.RFs(out4, filter_4, name='RF1', reduction=reduction)

        # 4, 16, 16, 64
        X3 = KL.Concatenate(name='main_con')([P.Up(out4, name='Up'), out3])
        X3 = KL.Dropout(0.5, name='main_Drop')(X3)
        X3 = P.RFs(X3, filter_4, name='RF2', reduction=reduction)

        # 8, 32, 32, 32
        X2 = KL.Concatenate(name='main_con1')([P.Up(P.Up(out4, name='Up1'), name='Up2'), P.Up(out3, name='Up3'), out2])
        X2 = KL.Dropout(0.5, name='main_Drop1')(X2)
        X2 = P.RFs(X2, filter_4, name='RF3', reduction=reduction)
        # print(X2)
        # X0 = KL.Concatenate()([out1, out0])
        # X0 = KL.Dropout(0.5, name='main_Drop2')(out1)
        # 16, 64, 64, 16
        X0 = out1
        X0 = KL.Conv3D(X0.shape[-1], 3, padding='same', strides=2, kernel_initializer=tf.keras.initializers.he_normal,
                       name='main_con2')(X0)
        X0 = P.RFs(X0, filter_4, name='RF4', reduction=reduction)

        out0_0 = out = P.PDC(X0, X2, X3, X4, filter_4, 'PDC')

        out1 = P.Block(out, 3, out.shape[-1] * 2, 'Seg')
        out1 = Conv3D(2, 1, activation='softmax',
                      kernel_initializer=tf.keras.initializers.he_normal)(out1)
        out_Seg = UpSampling3D(size=8)(out1)

        out0_0 = P.Up(out0_0, name='Up4')
        out0_0 = P.Block(out0_0, 3, filter_4, name='B')
        out0_0 = P.PDC_1(None, None, out0_0, out, filter_4, 'PDC1', out1[..., 0][..., None])
        out0_0 = KL.Concatenate(name='main_con_out')([out0_0, out0])
        out0_0 = P.Block(out0_0, 3, filter_4, name='B1')
        o = P.Up(out0_0, name='Up5')
        out0_1 = KL.Conv3D(2, 1, padding='same',
                           kernel_initializer=tf.keras.initializers.he_normal,
                           name='main_out1')(o)
        out0_1 = KL.Activation(activation='softmax', name='out1')(out0_1)

        model = Model([inputs, inputs_1], [out0_1, out_Seg])
        return model

    @staticmethod
    def SE(x, r=2):
        G_x = GlobalAveragePooling3D()(x)
        filters = x.shape[-1] / r
        F_x = Dense(filters)(G_x)
        F_x = LeakyReLU(0.1)(F_x)
        F_x = Dense(filters * r)(F_x)
        F_x = Activation('sigmoid')(F_x)
        F_x = Lambda(lambda x: K.expand_dims(x, 1))(F_x)
        F_x = Lambda(lambda x: K.expand_dims(x, 2))(F_x)
        F_x = Lambda(lambda x: K.expand_dims(x, 3))(F_x)
        x = x * F_x
        return x

    @staticmethod
    def CE_block(inputs, name):
        filters = inputs.shape[-1] / 4
        x0_1 = x0 = P.Block(inputs, 1, filters, name + 'CE_1')
        x0 = P.Block(x0, (3, 1, 3), filters, name + 'CE_2')
        x0 = P.Block(x0, (3, 3, 1), filters, name + 'CE_3')
        x0 = P.Block(x0, 3, filters, is_dilated=True, rate=(3, 3, 3), name=name + 'CE_4')
        x0 = Add()([x0, x0_1])

        x1_1 = x1 = P.Block(inputs, 1, filters, name + 'CE_5')
        x1 = P.Block(x1, (5, 1, 5), filters, name + 'CE_6')
        x1 = P.Block(x1, (5, 5, 1), filters, name + 'CE_7')
        x1 = P.Block(x1, 5, filters, is_dilated=True, rate=(5, 5, 5), name=name + 'CE_8')
        x1 = Add()([x1, x1_1])

        x2_1 = x2 = P.Block(inputs, 1, filters, name + 'CE_9')
        x2 = P.Block(x2, (7, 1, 7), filters, name + 'CE_10')
        x2 = P.Block(x2, (7, 7, 1), filters, name + 'CE_11')
        x2 = P.Block(x2, 7, filters, is_dilated=True, rate=(7, 7, 7), name=name + 'CE_12')
        x2 = Add()([x2, x2_1])

        x3 = P.Block(inputs, 1, filters, name + 'CE_13')
        # filters *= 4

        x = Concatenate()([x0, x1, x2, x3])
        x = P.Block(x, 3, filters * 4, name + 'CE_14')
        return x

    @staticmethod
    class weights(Layer):
        def __init__(self):
            super(P.weights, self).__init__()

        def build(self, input_shape):
            self.gama = self.add_weight(
                shape=1,
                initializer=tf.keras.initializers.ones
            )

        def call(self, inputs, **kwargs):
            X = inputs * self.gama
            return X

    @staticmethod
    def FM(x, x_feature, y, name):
        if x_feature.shape[1] != y.shape[1]:
            Higher_level = UpSampling3D()(x_feature)
        else:
            Higher_level = x_feature
        # Higher_level = min_max()(Higher_level)
        F_ba = Multiply()([(1 - Higher_level), y])
        F_fa = Multiply()([Higher_level, y])
        F_fnd = P.CE_block(F_ba, name=name + 'FM1')
        F_fnd = P.weights()(F_fnd)
        F_fpd = P.CE_block(F_fa, name=name + 'FM2')
        F_fpd = P.weights()(F_fpd)
        F_h = Conv3D(y.shape[-1], 3, padding='same', kernel_initializer=tf.keras.initializers.he_normal)(x)
        F_h = BatchNormalization()(F_h)
        F_h = LeakyReLU(.1)(F_h)

        if x_feature.shape[1] != y.shape[1]:
            F_h = UpSampling3D()(F_h)
        # F_h = UpSampling3D()(F_h)
        F_up = Add()([F_h, -F_fpd])
        F_up = Conv3D(y.shape[-1], 3, padding='same', kernel_initializer=tf.keras.initializers.he_normal)(F_up)
        F_up = BatchNormalization()(F_up)
        F_up = LeakyReLU(.1)(F_up)

        F_r = Add()([F_up, F_fnd])
        F_r = Conv3D(F_r.shape[-1], 3, padding='same', kernel_initializer=tf.keras.initializers.he_normal)(F_r)
        F_r = BatchNormalization()(F_r)
        F_r = LeakyReLU(.1)(F_r)

        return F_r


if __name__ == '__main__':
    model = P
    model = model.build()
    model.summary()
    print(model.output)
    # x = np.linspace(0, 1, 10)
    # print(x.shape)
    # for i in np.linspace(0, 1, 10):
    #     print(i)
