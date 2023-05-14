from keras import Model
from keras import layers
from keras.layers import *


class ResBlock(Model):
    def __init__(self, filters, stride=1):
        super(ResBlock, self).__init__(name="ResBlock")
        self.flag = (stride != 1)
        self.conv_1 = Conv2D(filters=filters, kernel_size=(3, 3), strides=(stride, stride), padding="same")
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(filters=filters, kernel_size=(3, 3), padding="same")
        self.bn_2 = BatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn_3 = BatchNormalization()
            self.conv_3 = Conv2D(filters=filters, kernel_size=(1, 1), strides=(stride, stride))

    def call(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.bn_1(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.conv_2(x_1)
        x_1 = self.bn_2(x_1)
        if self.flag:
            x = self.conv_3(x)
            x = self.bn_3(x)
        x_1 = layers.add([x, x_1])
        x_1 = self.relu(x_1)
        return x_1


class ResNet34(Model):
    def __init__(self):
        super(ResNet34, self).__init__(name="ResNet34")
        self.conv_1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same")
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))

        self.conv_2_1 = ResBlock(filters=64)
        self.conv_2_2 = ResBlock(filters=64)
        self.conv_2_3 = ResBlock(filters=64)

        self.conv_3_1 = ResBlock(filters=128, stride=2)
        self.conv_3_2 = ResBlock(filters=128)
        self.conv_3_3 = ResBlock(filters=128)
        self.conv_3_4 = ResBlock(filters=128)

        self.conv_4_1 = ResBlock(filters=256, stride=2)
        self.conv_4_2 = ResBlock(filters=256)
        self.conv_4_3 = ResBlock(filters=256)
        self.conv_4_4 = ResBlock(filters=256)
        self.conv_4_5 = ResBlock(filters=256)
        self.conv_4_6 = ResBlock(filters=256)

        self.conv_5_1 = ResBlock(filters=512, stride=2)
        self.conv_5_2 = ResBlock(filters=512)
        self.conv_5_3 = ResBlock(filters=512)

        self.pool = GlobalAveragePooling2D()
        self.fc_1 = Dense(512, activation="relu")
        self.dp_1 = Dropout(0.5)
        self.fc_2 = Dense(512, activation="relu")
        self.dp_2 = Dropout(0.5)
        self.fc_3 = Dense(64)

    def call(self, x):
        x = self.conv_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp(x)

        x = self.conv_2_1(x)
        x = self.conv_2_2(x)
        x = self.conv_2_3(x)

        x = self.conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.conv_3_3(x)
        x = self.conv_3_4(x)

        x = self.conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.conv_4_3(x)
        x = self.conv_4_4(x)
        x = self.conv_4_5(x)
        x = self.conv_4_6(x)

        x = self.conv_5_1(x)
        x = self.conv_5_2(x)
        x = self.conv_5_3(x)

        x = self.pool(x)
        x = self.fc_1(x)
        x = self.dp_1(x)
        x = self.fc_2(x)
        x = self.dp_2(x)
        x = self.fc_3(x)
        return x



