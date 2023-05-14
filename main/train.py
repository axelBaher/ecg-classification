import model as m


EPOCHS = 1
NUM_CLASSES = 8
BATCH_SIZE = 128


def train_loop(model):
    training_epoch = 1
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


def main():
    model = m.ResNet34()
    model.build(input_shape=(1, 128, 128, 1))
    train_loop(model)


if __name__ == "__main__":
    main()
