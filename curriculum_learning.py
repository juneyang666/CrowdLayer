from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Dense, merge, Reshape, Permute, Multiply, Dot,dot, Concatenate, Add
from keras.layers import Input
from keras import backend as K
from keras.engine.topology import Layer
import keras as keras

# packages for learning from crowds
from crowd_layer.crowd_layers import CrowdsClassification, MaskedMultiCrossEntropy, CrowdsClassificationSModel, \
    CrowdsClassificationCModelSingleWeight, CrowdsClassificationCModel, MaskedMultiCrossEntropyCosSim, \
    MaskedMultiCrossEntropyBaseChannel, MaskedMultiCrossEntropyBaseChannelConst, CrowdsClassificationSModelChannelMatrix, \
    MaskedMultiCrossEntropyCurriculumChannelMatrix
from crowd_layer.crowd_aggregators import CrowdsCategoricalAggregator


# prevent tensorflow from allocating the entire GPU memory at once
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

model = 'B'
NUM_RUNS = 30
DATA_PATH = "/Users/yangyajing/Documents/noisy_dataset/LabelMe/prepared/"
N_CLASSES = 8
BATCH_SIZE = 64
N_EPOCHS = 20
W = 0


# Prepare data
def load_data(filename):
    f = open(filename, 'rb')
    data = np.load(f)
    f.close()
    return data

def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

print("\nLoading train data...")

# images processed by VGG16
data_train_vgg16 = load_data(DATA_PATH+"data_train_vgg16.npy")
print(data_train_vgg16.shape)

# ground truth labels
labels_train = load_data(DATA_PATH+"labels_train.npy")
print(labels_train.shape)

# labels obtained from majority voting
labels_train_mv = load_data(DATA_PATH+"labels_train_mv.npy")
print(labels_train_mv.shape)

# labels obtained by using the approach by Dawid and Skene
labels_train_ds = load_data(DATA_PATH+"labels_train_DS.npy")
print(labels_train_ds.shape)

# data from Amazon Mechanical Turk
print("\nLoading AMT data...")
answers = load_data(DATA_PATH+"answers.npy")
print(answers.shape)
N_ANNOT = answers.shape[1]
print("N_CLASSES:", N_CLASSES)
print("N_ANNOT:", N_ANNOT)

# load test data
print("\nLoading test data...")

# images processed by VGG16
data_test_vgg16 = load_data(DATA_PATH+"data_test_vgg16.npy")
print(data_test_vgg16.shape)

# test labels
labels_test = load_data(DATA_PATH+"labels_test.npy")
print(labels_test.shape)

print("\nLoading validation data...")
# images processed by VGG16
data_valid_vgg16 = load_data(DATA_PATH+"data_valid_vgg16.npy")
print(data_valid_vgg16.shape)

# validation labels
labels_valid = load_data(DATA_PATH+"labels_valid.npy")
print(labels_valid.shape)

print("\nConverting to one-hot encoding...")
labels_train_bin = one_hot(labels_train, N_CLASSES)
print(labels_train_bin.shape)
labels_train_mv_bin = one_hot(labels_train_mv, N_CLASSES)
print(labels_train_mv_bin.shape)
labels_train_ds_bin = one_hot(labels_train_ds, N_CLASSES)
print(labels_train_ds_bin.shape)
labels_test_bin = one_hot(labels_test, N_CLASSES)
print(labels_test_bin.shape)
labels_valid_bin = one_hot(labels_valid, N_CLASSES)
print(labels_valid_bin.shape)


answers_bin_missings = []
for i in range(len(answers)):
    row = []
    for r in range(N_ANNOT):
        if answers[i,r] == -1:
            row.append(-1 * np.ones(N_CLASSES))
        else:
            row.append(one_hot(answers[i,r], N_CLASSES)[0,:])
    answers_bin_missings.append(row)
answers_bin_missings = np.array(answers_bin_missings).swapaxes(1,2)

answers_test_bin_missings = np.zeros((len(labels_test), N_CLASSES))
answers_test_bin_missings[np.arange(len(labels_test)), labels_test] = 1
answers_test_bin_missings = np.repeat(answers_test_bin_missings.reshape([len(labels_test),N_CLASSES,1]), N_ANNOT, axis=2)

answers_valid_bin_missings = np.zeros((len(labels_valid), N_CLASSES))
answers_valid_bin_missings[np.arange(len(labels_valid)), labels_valid] = 1
answers_valid_bin_missings = np.repeat(answers_valid_bin_missings.reshape([len(labels_valid),N_CLASSES,1]), N_ANNOT, axis=2)


# Build model
def eval(model,y_test):
    print('Test dataset results: ')
    print(dict(zip(model.metrics_names,model.evaluate(data_test_vgg16,y_test, verbose=False))))


def print_trace(model):

    channel_matrix = model.get_weights()[-1]
    channel_matrix_trace = tf.trace(K.permute_dimensions(channel_matrix, [2, 0, 1]))
    print(K.eval(channel_matrix_trace))
    return


def print_single_loss(model):
    import matplotlib.pyplot as plt

    # list all data in history
    print(model.history.keys())
    # summarize history for accuracy
    plt.plot(model.history['baseline_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.show()
    # summarize history for loss
    plt.plot(model.history['baseline_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


def print_history(df, title):
    import matplotlib.pyplot as plt
    # Make a data frame
    df['x'] = range(df.shape[0])

    # style
    plt.style.use('seaborn-darkgrid')

    # create a color palette
    palette = plt.get_cmap('Set1')

    # multiple line plot
    num = 0
    for column in df.drop('x', axis=1):
        num += 1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)

    # Add legend
    plt.legend(loc=2, ncol=2)

    # Add titles
    plt.title(title, loc='left', fontsize=12, fontweight=0, color='orange')
    plt.xlabel("Time")
    plt.ylabel("Score")
    plt.savefig(title+'.png')


acc_df = pd.DataFrame()
loss_df = pd.DataFrame()

for b in [0, 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.07, 0.05, 0.03, 0.01,
          0.007, 0.005, 0.001]:

    hidden_layers = Sequential()
    hidden_layers.add(Flatten(input_shape=data_train_vgg16.shape[1:]))
    hidden_layers.add(Dense(128, activation='relu'))
    hidden_layers.add(Dropout(0.5))
    hidden_layers.add(Dense(64, activation='relu'))
    hidden_layers.add(Dropout(0.5))

    train_inputs = Input(shape=(data_train_vgg16.shape[1:]))
    last_hidden = hidden_layers(train_inputs)
    baseline_output = Dense(N_CLASSES, activation='softmax', name='baseline')(last_hidden)

    if model == 'B':
        channel_layer = CrowdsClassificationSModelChannelMatrix(N_CLASSES, N_ANNOT)
    elif model == 'SW+B':
        channel_layer = CrowdsClassificationCModelSingleWeight(N_CLASSES, N_ANNOT, conn_type="MW")
    elif model == 'MW+B':
        channel_layer = CrowdsClassificationCModel(N_CLASSES, N_ANNOT, conn_type="MW")
    else:
        raise Exception("Unknown type for CrowdsClassification layer!")

    channeled_output = channel_layer([last_hidden, baseline_output])

    crowd_model = Model(inputs=train_inputs, outputs=[channeled_output, baseline_output])

    curr_loss = MaskedMultiCrossEntropyCurriculumChannelMatrix(crowd_model, 1, b).loss

    # compile model with masked loss and train
    crowd_model.compile(optimizer='adam',
                         loss=[curr_loss, 'categorical_crossentropy'],
                         loss_weights=[1, 0],
                         metrics=['accuracy']
                        )

    eval(crowd_model, y_test=[answers_test_bin_missings, labels_test_bin])
    print_trace(crowd_model)
    history = crowd_model.fit(data_train_vgg16, [answers_bin_missings, labels_train_bin], epochs=35, shuffle=True,
                              batch_size=BATCH_SIZE, verbose=1)
    print_trace(crowd_model)
    eval(crowd_model, y_test=[answers_test_bin_missings, labels_test_bin])
    acc_df.loc[:, str(b)] = history.history['baseline_acc']
    loss_df.loc[:, str(b)] = history.history['baseline_loss']

print_history(acc_df, 'accuracy')
print_history(loss_df, 'loss')
