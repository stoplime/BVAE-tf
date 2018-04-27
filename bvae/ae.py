'''
vae.py
contains the setup for autoencoders.

created by shadySource

THE UNLICENSE
'''
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K

class AutoEncoder(object):
    def __init__(self, encoderArchitecture, 
                 decoderArchitecture):

        self.encoder = encoderArchitecture.model
        self.decoder = decoderArchitecture.model

        self.ae = Model(self.encoder.inputs, self.decoder(self.encoder.outputs))

def test():
    import os
    import numpy as np
    from PIL import Image
    from tensorflow.python.keras.preprocessing.image import load_img

    import models

    inputShape = (256, 256, 3)
    batchSize = 20
    latentSize = 100

    img = load_img(os.path.join('..','images', 'img.jpg'), target_size=inputShape[:-1])
    img.show()

    img = np.array(img, dtype=np.float32) / 255 - 0.5
    img = np.array([img]*batchSize) # make fake batches to improve GPU utilization

    # This is how you build the autoencoder
    encoder = models.BetaEncoder(inputShape, batchSize, latentSize, 'bvae', beta=69, capacity=15, randomSample=True)
    decoder = models.BetaDecoder(inputShape, batchSize, latentSize)
    bvae = AutoEncoder(encoder, decoder)

    bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')
    while True:
        bvae.ae.fit(img, img,
                    epochs=100,
                    batch_size=batchSize)
        
        # example retrieving the latent vector
        latentVec = bvae.encoder.predict(img)[0]
        print(latentVec)

        pred = bvae.ae.predict(img) # get the reconstructed image
        pred[pred > 0.5] = 0.5 # clean it up a bit
        pred[pred < -0.5] = -0.5
        pred = np.uint8((pred + 0.5)* 255) # convert to regular image values

        pred = Image.fromarray(pred[0])
        pred.show() # display popup

def test2():
    import os
    import numpy as np
    from PIL import Image
    from tensorflow.python.keras.preprocessing.image import load_img
    import cv2
    import models
    import random

    inputShape = (64, 64, 1)
    batchSize = 32
    latentSize = 10
    episodes = 10000
    verbose = 1

    loadFolder = 'imageNet2'
    loadFile = 'PtBetaEncoder-32px-128l-1000e'
    load = False
    saveFolder = 'Dsprite1'
    saveFile = 'PtDarkNet19-bvae-100-10l-20c-64px-10000e'
    save = True

    useDsprites = True

    # C:\Users\slani\Documents\GitHub\montazuma\dataset\0000001.png
    # C:\Users\slani\Documents\GitHub\montazuma\dataset\1281149.png
    # dataPath = os.path.join('..', '..', 'dataset', 'train_32x32')
    dataPath = os.path.join('..', 'dataset')

    loadFolderPath = os.path.join('..', 'save', loadFolder)
    saveFolderPath = os.path.join('..', 'save', saveFolder)
    loadPath = os.path.join(loadFolderPath, loadFile + '.h5')
    savePath = os.path.join(saveFolderPath, saveFile + '.h5')

    if useDsprites:
        print("loading images")
        data = np.load(os.path.join(dataPath, "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"))
        dataImages = data['imgs']
        dataImages = np.swapaxes(dataImages,0,2)
        print("dataImages.shape", dataImages.shape)
        print("Images Aquired")
    
    if not os.path.exists(saveFolderPath):
        os.makedirs(saveFolderPath)

    # This is how you build the autoencoder
    encoder = models.Darknet19Encoder(inputShape, batchSize, latentSize, 'bvae', beta=100, capacity=20, randomSample=True)
    decoder = models.Darknet19Decoder(inputShape, batchSize, latentSize)
    bvae = AutoEncoder(encoder, decoder)

    bvae.ae.compile(optimizer='adam', loss='mean_absolute_error')

    if load:
        bvae.ae.load_weights(loadPath)

    for epoch in range(episodes):
        imgs = []
        for _batch in range(batchSize):
            if useDsprites:
                imageNum = random.randrange(1, 737280)
                img = dataImages[:,:,imageNum]
                img = img*255
                imgs.append(img)
            else:
                imageNum = str(random.randrange(1, 1281150)).zfill(7)
                img = load_img(os.path.join(dataPath, imageNum+".png"), target_size=inputShape[:-1])
                imgs.append(np.array(img, dtype=np.uint8))

        
        if verbose == 1:
            batch_view = np.array(imgs, dtype=np.uint8)
            if useDsprites:
                batch_view = np.expand_dims(batch_view, axis=3)
            print("batch.shape", batch_view.shape)
            visualize_batch = np.concatenate((*batch_view,), axis=1)
            # visualize_batch = cv2.cvtColor(visualize_batch, cv2.COLOR_RGB2BGR)
            cv2.imshow("batch", visualize_batch)
            cv2.waitKey(1)
        
        batch = np.array(imgs, dtype=np.float32)
        if useDsprites:
            batch = np.expand_dims(batch, axis=3)
        batch = batch / 255 - 0.5

        bvae.ae.fit(batch, batch,
                    epochs=100,
                    batch_size=batchSize)
        
        # example retrieving the latent vector
        if verbose == 1:
            latentVec = bvae.encoder.predict(batch)[0]
            print(latentVec)

        print("episode: {}/{}".format(epoch+1, episodes))
        if save:
            bvae.ae.save_weights(savePath)

        if verbose == 1:
            pred = bvae.ae.predict(batch) # get the reconstructed image
            print("pred.shape", pred.shape)
            pred[pred > 0.5] = 0.5 # clean it up a bit
            pred[pred < -0.5] = -0.5
            pred = np.uint8((pred + 0.5)* 255) # convert to regular image values

            visualize_pred = np.concatenate((*pred,), axis=1)
            if not useDsprites:
                visualize_pred = cv2.cvtColor(visualize_pred, cv2.COLOR_RGB2BGR)
            visualize_both = np.concatenate((visualize_batch, visualize_pred), axis=0)

            cv2.imwrite(os.path.join(saveFolderPath, "sample_{}.png".format(str(epoch).zfill(7))), visualize_both)
            cv2.imshow("prediction", visualize_both)
            cv2.waitKey(1)

            # pred = Image.fromarray(pred[0])
            # pred.show() # display popup

if __name__ == "__main__":
    test2()
