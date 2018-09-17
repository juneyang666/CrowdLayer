import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiCrossEntropy, CrowdsClassificationSModel, \
    CrowdsClassificationCModelSingleWeight, CrowdsClassificationCModel
from keras.models import Sequential, Model



def datagen(train_data_dir, valid_data_dir, sz=228, batch_size=64):

    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = \
        train_datagen.flow_from_directory(train_data_dir,
                                          target_size=(sz, sz),
                                          batch_size=batch_size)

    validation_generator = \
        test_datagen.flow_from_directory(valid_data_dir,
                                         shuffle=False,
                                         target_size=(sz, sz),
                                         batch_size=batch_size)

    return train_generator, validation_generator

def create_model(N_CLASSES=8, N_ANNOT=59):
    base_model = VGG16(weights='imagenet', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    last_hidden = Dropout(0.5)(x)

    baseline_output = Dense(N_CLASSES, activation='softmax', name='baseline')(last_hidden)
    # channeled_output = CrowdsClassificationSModel(N_CLASSES, N_ANNOT)([last_hidden, baseline_output])

    # crowd_model = Model(inputs=base_model.input, outputs=[channeled_output, baseline_output])
    crowd_model = Model(inputs=base_model.input, outputs=baseline_output)

    # loss = MaskedMultiCrossEntropy().loss

    for layer in base_model.layers:
        layer.trainable = False

    # compile model with masked loss and train
    crowd_model.compile(optimizer='adam',
                        loss= 'categorical_crossentropy',
                        # loss_weights=[1, 0],
                        metrics=['accuracy']
                        )

    return crowd_model


def unfreeze_model(model):
    split_at = 140
    for layer in model.layers[:split_at]: layer.trainable = False
    for layer in model.layers[split_at:]: layer.trainable = True

    # loss = MaskedMultiCrossEntropy().loss
    model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        # loss_weights=[1, 0],
                        metrics=['accuracy']
                        )
    return model


def eval(model, x_test_generator, y_test):
    print('Test dataset results: ')
    print(dict(zip(model.metrics_names,model.evaluate(x_test_generator,y_test, verbose=False))))


def main():
    train_data_dir = '/Users/yangyajing/Documents/noisy_dataset/LabelMe/train'
    valid_data_dir = '/Users/yangyajing/Documents/noisy_dataset/LabelMe/valid'
    test_data_dir = '/Users/yangyajing/Documents/noisy_dataset/LabelMe/test'
    batch_size = 64

    train_generator, validation_generator = datagen(train_data_dir, valid_data_dir, sz=228, batch_size=64)
    crowd_model = create_model()

    crowd_model.fit_generator(train_generator,
                              train_generator.n//batch_size,
                              epochs=3,
                              workers=2,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.n//batch_size)

    crowd_model = unfreeze_model(crowd_model)

    crowd_model.fit_generator(train_generator,
                              train_generator.n//batch_size,
                              epochs=3,
                              workers=2,
                              validation_data=validation_generator,
                              validation_steps=validation_generator.n//batch_size)


if __name__ == "__main__":
    main()

