import keras
from awx_core.layers import *
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve




def classify(train, test, val, epoch, indices):
    section_num, class_num, subclass_num = indices


    clf = keras.models.Sequential([
    keras.layers.Dense(
        500,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l1_l2(l2=0, l1=0),
        name='dense_1'
    ),
    keras.layers.GaussianNoise(0.1),
    keras.layers.Dense(
        500,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l1_l2(l2=0, l1=0),
        name='dense_2'
    ),
    keras.layers.GaussianNoise(0.1),
    keras.layers.Dense(
        500,
        activation='tanh',
        kernel_regularizer=keras.regularizers.l1_l2(l2=0, l1=0),
        name='dense_3'
    ),
    keras.layers.GaussianNoise(0.1),
    AWX(
        A=train.A, 
        n_norm=1, 
        activation='sigmoid', 
        kernel_regularizer=keras.regularizers.l1_l2(l1=0, l2=1e-6), 
        name='AWX'
    )
    ])

    clf.compile(
        keras.optimizers.Adam(lr=1e-5),
        loss = 'binary_crossentropy',
        metrics = ['binary_crossentropy']
    )

    clf.fit(
        train.X,
        train.Y,

        validation_data=[
            val.X,
            val.Y,
        ],
        epochs=epoch, 
        batch_size=32,
        initial_epoch=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, monitor='val_loss', mode='auto', ),
        ],
        verbose=2
    )

    Y_prob_train = clf.predict(train.X)
    Y_prob_test = clf.predict(test.X)

    #Y_pred = (Y_prob_test > 0.5) 
    Y_pred = np.where(Y_prob_test >= 0.5, 1, 0)
    Y_pred_train = np.where(Y_prob_train >= 0.5, 1, 0)
    print('Y_pred', Y_pred)
    #print('test.Y.sum(0)!=0',test.Y.sum(0)!=0, 'len(test.Y.sum(0))', len(test.Y.sum(0)))
    #print('test.A.sum(1)==0',test.A.sum(1)==0, 'len(test.A.sum(1))', len(test.A.sum(1)))


    cases = ['SECTION', 'CLASS', 'SUBCLASS']
    case_indices = [list(range(1,1+section_num)), list(range(1+section_num,1+section_num+class_num)),
                    list(range(1+section_num+class_num,Y_pred.shape[1]))]

    print('Original:')
    print('Train:')
    print('     average_precision_score', average_precision_score(train.Y[:,train.Y.sum(0)!=0], Y_pred_train[:,train.Y.sum(0)!=0], average='micro'))
    print('     f1_score', f1_score(train.Y[:,train.Y.sum(0)!=0], Y_pred_train[:,train.Y.sum(0)!=0], average='weighted'))
    print('Test:')
    print('     average_precision_score', average_precision_score(test.Y[:,test.Y.sum(0)!=0], Y_pred[:,test.Y.sum(0)!=0], average='micro'))
    print('     f1_score', f1_score(test.Y[:,test.Y.sum(0)!=0], Y_pred[:,test.Y.sum(0)!=0], average='weighted'))


    for i, name in enumerate(cases):
        index = case_indices[i]
        print('\n\nFor case', name)
        print('Training dataset:')
        print('     average_precision_score', average_precision_score(train.Y[:,index], Y_pred_train[:,index], average='micro'))
        print('     f1_score', f1_score(train.Y[:,index], Y_pred_train[:,index], average='weighted'))
        
        print('\nTesting dataset:')
        print('     average_precision_score', average_precision_score(test.Y[:,index], Y_pred[:,index], average='micro'))
        print('     f1_score', f1_score(test.Y[:,index], Y_pred[:,index], average='weighted'))

    return clf



