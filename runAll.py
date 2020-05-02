from dataset.prepareDataset import prepareDataset
from dataset.generateTfrecord import generateTfrecord
from mtcnn import trainPnet, trainRnet
from predict import predict

# prepareDataset(True, size=24)
# generateTfrecord(size=24, rnetFlag=True)
# trainPnet(type="pnetBoxRegression", loadModelFlag=True, size=12, epochs=20)
# trainPnet(type="pnetClassification", loadModelFlag=True, size=12, epochs=5)
trainRnet(type='rnetBoxRegression', loadModelFlag=True, size=24, epochs=1)
# predict(kernelSize=12, stride=2)