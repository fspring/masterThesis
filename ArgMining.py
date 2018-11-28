import numpy as np
import tensorflow as tf
import random as rn
import os
import datetime
import re
import random
import argparse
import string
import keras

from sklearn.model_selection import KFold

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Layer, Dense, Embedding, LSTM, Bidirectional
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping

from utils import bool_flag

from crf import CRF

class Pentanh(Layer):

    def __init__(self, **kwargs):
        super(Pentanh, self).__init__(**kwargs)
        self.supports_masking = True
        self.__name__ = 'pentanh'
    
    def call(self, inputs): 
        return K.switch(K.greater(inputs,0), K.tanh(inputs), 0.25 * K.tanh(inputs))
    
    def get_config(self): 
        return super(Pentanh, self).get_config()
    
    def compute_output_shape(self, input_shape): 
        return input_shape

keras.utils.generic_utils.get_custom_objects().update({'pentanh': Pentanh()})

class TextReader:

    def __init__(self, baseDirectory):
        self.contents = []
        self.baseDirectory = baseDirectory

    def processText(self, fileName):
        file = open(fileName, "r", encoding='utf8')
        textContents = file.read()
        textContents = re.sub(r'\n', r' ', textContents)
        self.contents.append(textContents)

    def readTexts(self):
        fileList = os.listdir(self.baseDirectory)
        for file in fileList:
            fileName = self.baseDirectory + '/' + file
            self.processText(fileName)


class TagProcessor:
    def __init__(self, baseDirectory, nArgTag):
        self.contents = []
        self.num_tags = 3
        self.baseDirectory = baseDirectory
        self.nArgTag = nArgTag

        self.numNArg = 0
        self.numClaim = 0
        self.numPremise = 0

    def encode(self, tagList):
        encodedTags = []
        for tags in tagList:
            newTags = []
            for tag in tags:
                if tag == 0:
                    newTags.append(self.nArgTag)
                else:
                    if tag == 1:
                        self.numPremise += 1
                    if tag == 2:
                        self.numNArg += 1
                    if tag == 3:
                        self.numClaim += 1
                    result = np.zeros(self.num_tags)
                    result[tag - 1] = 1
                    newTags.append(list(map(int, result)))
            encodedTags.append(newTags)
        return encodedTags

    def processTags(self, fileName):
        file = open(fileName, "r", encoding='utf8')
        TagContents = file.read()
        TagContents = re.sub(r'\n', r' ', TagContents)
        self.contents.append(TagContents)

    def readTags(self):
        fileList = os.listdir(self.baseDirectory)
        for file in fileList:
            fileName = self.baseDirectory + '/' + file
            self.processTags(fileName)

class SequenceCreator:
    englishTextsMaxSize = 735

    def __init__(self, allTexts, textsToEval, maxlen):
        self.sequences = []
        self.maxlen = 0
        self.allTexts = allTexts
        self.textsToEval = textsToEval
        self.maxlen = maxlen

    def normalizeSequences(self):
        maxlen = 0
        newSequences = []
        for sequence in self.sequences:
            if len(sequence) > maxlen:
                maxlen = len(sequence)
        if maxlen:
            self.maxlen = maxlen
        else:
            self.maxlen = self.englishTextsMaxSize

        for sequence in self.sequences:
            while len(sequence) < self.maxlen:
                sequence.append(0)
            newSequences.append(sequence)
        self.sequences = newSequences

    def createSequences(self):
        token = Tokenizer(filters='')
        token.fit_on_texts(self.allTexts.contents)
        self.wordIndex = token.word_index
        self.sequences = token.texts_to_sequences(self.textsToEval.contents)
        self.normalizeSequences()

    def transformText(self, text, textList):
        token = Tokenizer(filters='')
        token.fit_on_texts(textList.contents)
        sequence = token.texts_to_sequences([text])
        sequences = []

        for text in sequence:
            while len(text) < self.maxlen:
                text.append(0)
            sequences.append(text)
            sequences.append(text)

        return sequences


class NeuralTrainer:
    embedding_size = 300
    hidden_size = 100

    def __init__(self, maxlen, num_tags, wordIndex, embeddings, textsToEval, dumpPath):
        self.sequences = []
        self.maxlen = maxlen
        self.max_features = len(wordIndex)
        self.num_tags = num_tags
        self.wordIndex = wordIndex
        self.embeddings = embeddings
        self.textsToEval = textsToEval
        self.dumpPath = dumpPath

    def decodeTags(self, tags):
        newtags = []
        for tag in tags:
            newtag = np.argmax(tag)
            newtags.append(newtag)
        return newtags

    def createEmbeddings(self, word_index, embeddings):
        embeddings_index = {}
        path = 'Embeddings/' + embeddings + '.txt'
        f = open(path, "r", encoding='utf8')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()

        embedding_matrix = np.zeros((self.max_features + 1, self.embedding_size))
        for word, i in word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def createModel(self):
        embeddingMatrix = self.createEmbeddings(self.wordIndex, self.embeddings)
        model = Sequential()

        model.add(
            Embedding(self.max_features + 1, self.embedding_size, weights=[embeddingMatrix], input_length=self.maxlen,
                      trainable=False, mask_zero=True))

        model.add(TimeDistributed(Dense(self.hidden_size, activation='relu')))
        model.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, activation='pentanh', recurrent_activation='pentanh')))
        model.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, activation='pentanh', recurrent_activation='pentanh')))
        model.add(TimeDistributed(Dense(20, activation='relu')))

        crf = CRF(self.num_tags, sparse_target=False, learn_mode='join', test_mode='viterbi')
        model.add(crf)
        model.compile(optimizer='adam', loss=crf.loss_function, metrics=[crf.accuracy])

        return model

    def trainModel(self, x_train, y_train, x_test, y_test, unencodedY, testSet):
        model = self.createModel()
        monitor = EarlyStopping(monitor='loss', min_delta=0.001, patience=5, verbose=1, mode='auto')
        model.fit(x_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[monitor])
        scores = model.evaluate(x_test, y_test, batch_size=8, verbose=1)
        y_pred = model.predict(x_test)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

        self.printEvaluatedTexts(x_test, y_pred, testSet, self.textsToEval, self.dumpPath)
        spanEvalAt1 = self.spanEval(y_pred, unencodedY, 1.0)
        spanEvalAt075 = self.spanEval(y_pred, unencodedY, 0.75)
        spanEvalAt050 = self.spanEval(y_pred, unencodedY, 0.50)
        tagEval = self.tagEval(y_pred, unencodedY)

        return [scores[1], tagEval, spanEvalAt1, spanEvalAt075, spanEvalAt050]

    def crossValidate(self, X, Y, additionalX, additionalY, unencodedY):
        seed = 42
        n_folds = 10
        foldNumber  = 1

        kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        cvscores = [0, [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0]]
        for train, test in kfold.split(X, unencodedY):
            print('Fold')
            print(foldNumber)
            xTrainSet = []
            yTrainSet = []
            xTestSet = []
            yTestSet = []
            unencodedYSet = []
            for trainIndex in train:
                xTrainSet.append(X[trainIndex])
                yTrainSet.append(Y[trainIndex])
            for testIndex in test:
                xTestSet.append(X[testIndex])
                yTestSet.append(Y[testIndex])
                unencodedYSet.append(unencodedY[testIndex])

            xTrainSet = xTrainSet + additionalX
            yTrainSet = yTrainSet + additionalY
            scores = self.trainModel(xTrainSet, yTrainSet, xTestSet, yTestSet, unencodedYSet, test)
            cvscores = self.handleScores(cvscores, scores, n_folds)
            foldNumber += 1
        print('Average results for the ten folds:')
        self.prettyPrintResults(cvscores)
        return cvscores

    def handleScores(self, oldScores, newScores, nFolds):
        newAccuracy = oldScores[0] + (newScores[0] / nFolds)
        newTagScores = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        newSpanAt1Scores = [0, 0, 0, 0, 0, 0, 0]
        newSpanAt075Scores = [0, 0, 0, 0, 0, 0, 0]
        newSpanAt050Scores = [0, 0, 0, 0, 0, 0, 0]

        for i in range(0, 3):
            for j in range(0, 3):
                newTagScores[i][j] = oldScores[1][i][j] + (newScores[1][i][j] / nFolds)
        for j in range(0, 7):
            newSpanAt1Scores[j] = oldScores[2][j] + (newScores[2][j] / nFolds)
        for j in range(0, 7):
            newSpanAt075Scores[j] = oldScores[3][j] + (newScores[3][j] / nFolds)
        for j in range(0, 7):
            newSpanAt050Scores[j] = oldScores[4][j] + (newScores[4][j] / nFolds)

        return [newAccuracy, newTagScores, newSpanAt1Scores, newSpanAt075Scores, newSpanAt050Scores]

    def tagEval(self, y_pred, unencodedY, ):
        i = 0
        precision = []
        recall = []
        f1 = []
        accuracy = []
        for result in y_pred:
            sequenceLength = len(np.trim_zeros(unencodedY[i]))
            result = np.resize(result, (sequenceLength, self.num_tags))
            classes = np.argmax(result, axis=1)
            accuracy.append(accuracy_score(np.trim_zeros(unencodedY[i]), np.add(classes, 1)))
            scores = precision_recall_fscore_support(np.trim_zeros(unencodedY[i]), np.add(classes, 1))
            precision.append(np.pad(scores[0], (0,(3 - len(scores[0]))), 'constant'))
            recall.append(np.pad(scores[1], (0,(3 - len(scores[0]))), 'constant'))
            f1.append(np.pad(scores[2], (0,(3 - len(scores[0]))), 'constant'))
            i += 1
        print("Accuracy = %.3f%% (+/- %.3f%%)" % (np.mean(accuracy), np.std(accuracy)))
        precision = self.prettyPrintScore(precision, 'Precision')
        recall = self.prettyPrintScore(recall, 'Recall')
        f1 = self.prettyPrintScore(f1, 'F1')

        return [precision, recall, f1]

    def prettyPrintScore(self, score, scoreName):
        print(scoreName)
        numTexts = len(score)
        scoreFirstClass = 0
        scoreSecondClass = 0
        scoreThirdClass = 0
        for scoreValue in score:
            scoreFirstClass += scoreValue[0]
            scoreSecondClass += scoreValue[1]
            scoreThirdClass += scoreValue[2]
        print('(i,premise)' + '(o,|)   ' + '(i,claim)')
        print(str(round(scoreFirstClass / numTexts, 3)) + '   ' + str(
            round(scoreSecondClass / numTexts, 3)) + '     ' + str(round(scoreThirdClass / numTexts, 3)))

        return [round(scoreFirstClass / numTexts, 4), round(scoreSecondClass / numTexts, 4),
                round(scoreThirdClass / numTexts, 4)]

    def prettyPrintResults(self, scores):
        print('Accuracy - ' + str(round(scores[0], 4)))

        print('Accuracy at ' + str(1) + ' - ' + str(round(scores[2][0], 3)))
        print('Accuracy at ' + str(0.75) + ' - ' + str(round(scores[3][0], 3)))
        print('Accuracy at ' + str(0.5) + ' - ' + str(round(scores[4][0], 3)))

        print('Precision for premises at ' + str(1) + ' - ' + str(round(scores[2][1], 3)))
        print('Precision for claims at ' + str(1) + ' - ' + str(round(scores[2][2], 3)))
        print('Precision for premises at ' + str(0.75) + ' - ' + str(round(scores[3][1], 3)))
        print('Precision for claims at ' + str(0.75) + ' - ' + str(round(scores[3][2], 3)))
        print('Precision for premises at ' + str(0.5) + ' - ' + str(round(scores[4][1], 3)))
        print('Precision for claims at ' + str(0.5) + ' - ' + str(round(scores[4][2], 3)))

        print('Recall for premises at ' + str(1) + ' - ' + str(round(scores[2][3], 3)))
        print('Recall for claims at ' + str(1) + ' - ' + str(round(scores[2][4], 3)))
        print('Recall for premises at ' + str(0.75) + ' - ' + str(round(scores[3][3], 3)))
        print('Recall for claims at ' + str(0.75) + ' - ' + str(round(scores[3][4], 3)))
        print('Recall for premises at ' + str(0.5) + ' - ' + str(round(scores[4][3], 3)))
        print('Recall for claims at ' + str(0.5) + ' - ' + str(round(scores[4][4], 3)))

        print('F1 for premises at ' + str(1) + ' - ' + str(round(scores[2][5], 3)))
        print('F1 for claims at ' + str(1) + ' - ' + str(round(scores[2][6], 3)))
        print('F1 for premises at ' + str(0.75) + ' - ' + str(round(scores[3][5], 3)))
        print('F1 for claims at ' + str(0.75) + ' - ' + str(round(scores[3][6], 3)))
        print('F1 for premises at ' + str(0.5) + ' - ' + str(round(scores[4][5], 3)))
        print('F1 for claims at ' + str(0.5) + ' - ' + str(round(scores[4][6], 3)))

        print('Precision')
        print('(i,premise)' + '(o,|)   ' + '(i,claim)')
        print(str(round(scores[1][0][0], 3)) + '   ' + str(round(scores[1][0][1], 3)) + '     ' + str(
            round(scores[1][0][2], 3)))
        print('Recall')
        print('(i,premise)' + '(o,|)   ' + '(i,claim)')
        print(str(round(scores[1][1][0], 3)) + '   ' + str(round(scores[1][1][1], 3)) + '     ' + str(
            round(scores[1][1][2], 3)))
        print('F1')
        print('(i,premise)' + '(o,|)   ' + '(i,claim)')
        print(str(round(scores[1][2][0], 3)) + '   ' + str(round(scores[1][2][1], 3)) + '     ' + str(
            round(scores[1][2][2], 3)))

    def printEvaluatedTexts(self, x_test, y_pred, testSet, textList, dumpPath):
        invWordIndex = {v: k for k, v in self.wordIndex.items()}
        texts = []

        for i in range(0,len(x_test)):
            text = []
            tags = y_pred[i]
            j = 0
            for word in np.trim_zeros(x_test[i]):
                text.append([invWordIndex[word], np.argmax(tags[j])])
                j += 1
            texts.append(text)

        fileList = os.listdir(textList)
        filenames = []
        for file in fileList:
            fileName = dumpPath + '/' + file
            filenames.append(fileName)
        filenames = [filenames[x] for x in testSet]

        classes = ['(I,Premise)', '(O,|)', '(I,Claim)']
        for i in range(0, len(texts)):
            textFile = open(filenames[i], "w", encoding='utf-8')
            for token in texts[i]:
                textFile.write(u'' + token[0] + ' ' + classes[token[1]] + '\n')

    def spanCreator(self, unencodedY):
        spans = []
        for text in unencodedY:
            text = np.trim_zeros(text)
            textSpans = {}
            startPosition = 0
            currentPosition = 0
            lastTag = text[0]
            for tag in text:
                if tag != lastTag:
                    endPosition = currentPosition - 1
                    textSpans[startPosition] = endPosition
                    startPosition = currentPosition
                lastTag = tag
                currentPosition += 1
            endPosition = currentPosition - 1
            textSpans[startPosition] = endPosition
            spans.append(textSpans)

        return spans

    def trimSpans(self, predictedSpanStart, predictedSpanEnd, goldSpans):
        trimmedSpans = {}
        for goldSpanStart, goldSpanEnd in goldSpans[0].items():
            if ((predictedSpanStart >= goldSpanStart) and (predictedSpanEnd <= goldSpanEnd)):
                trimmedSpans[predictedSpanStart] = predictedSpanEnd
        return trimmedSpans

    def spanEval(self, y_pred, unencodedY, threshold):
        goldSpans = self.spanCreator(unencodedY)
        i = 0
        precision = [0, 0, 0]
        recall = [0, 0, 0]
        f1 = [0, 0, 0]
        predictedSpanTypes = [0, 0, 0]
        goldSpanTypes = [0, 0, 0]
        precisionCorrectSpans = [0, 0, 0]
        recallCorrectSpans = [0, 0, 0]
        for result in y_pred:
            sequenceLength = len(np.trim_zeros(unencodedY[i]))
            result = np.resize(result, (sequenceLength, self.num_tags))
            classes = np.argmax(result, axis=1)
            classes = np.add(classes, 1)

            for spanStart, spanEnd in goldSpans[i].items():
                goldSpanTypes[unencodedY[i][spanStart] - 1] += 1

            for spanStart, spanEnd in goldSpans[i].items():
                predicted = classes[spanStart:spanEnd + 1]
                possibleSpans = self.spanCreator([predicted])

                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    predictedSpanTypes[classes[spanStart + possibleSpanStart] - 1] += 1
                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    if (((possibleSpanEnd - possibleSpanStart + 1) >= ((spanEnd - spanStart + 1) * threshold))
                            and (classes[spanStart + possibleSpanStart] == unencodedY[i][
                                spanStart + possibleSpanStart])):
                        precisionCorrectSpans[classes[spanStart + possibleSpanStart] - 1] += 1
                        break
                for possibleSpanStart, possibleSpanEnd in possibleSpans[0].items():
                    if (((possibleSpanEnd - possibleSpanStart + 1) >= ((spanEnd - spanStart + 1) * threshold))
                            and (classes[spanStart + possibleSpanStart] == unencodedY[i][
                                spanStart + possibleSpanStart])):
                        recallCorrectSpans[classes[spanStart + possibleSpanStart] - 1] += 1
            i += 1

        accuracy = ((precisionCorrectSpans[0] + precisionCorrectSpans[2]) / (goldSpanTypes[0] + goldSpanTypes[2]))
        for i in range(0, 3):
            if (predictedSpanTypes[i] != 0):
                precision[i] = (precisionCorrectSpans[i] / predictedSpanTypes[i])
            if (goldSpanTypes[i] != 0):
                recall[i] = (recallCorrectSpans[i] / goldSpanTypes[i])
            if ((precision[i] + recall[i]) != 0):
                f1[i] = 2 * ((precision[i] * recall[i]) / (precision[i] + recall[i]))

        print('Accuracy at ' + str(threshold) + ' - ' + str(round(accuracy, 3)))
        print('Precision for premises at ' + str(threshold) + ' - ' + str(round(precision[0], 3)))
        print('Precision for claims at ' + str(threshold) + ' - ' + str(round(precision[2], 3)))
        print('Recall for premises at ' + str(threshold) + ' - ' + str(round(recall[0], 3)))
        print('Recall for claims at ' + str(threshold) + ' - ' + str(round(recall[2], 3)))
        print('F1 for premises at ' + str(threshold) + ' - ' + str(round(f1[0], 3)))
        print('F1 for claims at ' + str(threshold) + ' - ' + str(round(f1[2], 3)))

        return [round(accuracy, 4), round(precision[0], 4), round(precision[2], 4), round(recall[0], 4),
                round(recall[2], 4), round(f1[0], 4), round(f1[2], 4)]


def fullSequence(textDirectory, tagDirectory, addTexts, embeddings, dumpPath):
    commonTextDirectory = 'allTextsPunctuation'
    allTexts = TextReader(commonTextDirectory)
    allTexts.readTexts()

    texts = TextReader(textDirectory)
    texts.readTexts()
    if addTexts:
        textSequencer = SequenceCreator(allTexts, texts, False)
    else:
        textSequencer = SequenceCreator(allTexts, texts, True)
    textSequencer.createSequences()
    textSequences = textSequencer.sequences


    commonTagDirectory = 'allTagsPunctuation'
    allTags = TagProcessor(commonTagDirectory, [0, 1, 0])
    allTags.readTags()

    tags = TagProcessor(tagDirectory, [0, 1, 0])
    tags.readTags()
    if addTexts:
        tagSequencer = SequenceCreator(allTags, tags, False)
    else:
        tagSequencer = SequenceCreator(allTags, tags, True)
    tagSequencer.createSequences()
    unencodedTags = tagSequencer.sequences
    tagSequences = tags.encode(tagSequencer.sequences)

    model = NeuralTrainer(textSequencer.maxlen, tags.num_tags, textSequencer.wordIndex,
                          embeddings, textDirectory, dumpPath)
    startTime = datetime.datetime.now().replace(microsecond=0)

    if addTexts:
        englishTextsDirectory = 'essaysClaimsPremisesPunctuation/texts'
        englishTexts = TextReader(englishTextsDirectory)
        englishTexts.readTexts()
        englishTextSequencer = SequenceCreator(allTexts, englishTexts, False)
        englishTextSequencer.createSequences()
        englishTextSequences = englishTextSequencer.sequences

        englishTagDirectory = 'essaysClaimsPremisesPunctuation/tags'
        englishTags = TagProcessor(englishTagDirectory, [0, 1, 0])
        englishTags.readTags()
        englishTagSequencer = SequenceCreator(allTags, englishTags, False)
        englishTagSequencer.createSequences()
        englishTagSequences = englishTags.encode(englishTagSequencer.sequences)

        model.crossValidate(textSequences, tagSequences, englishTextSequences, englishTagSequences, unencodedTags)
    else:
        model.crossValidate(textSequences, tagSequences, [], [], unencodedTags)


    endTime = datetime.datetime.now().replace(microsecond=0)
    timeTaken = endTime - startTime

    print("Time elapsed:")
    print(timeTaken)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train and evaluate the argumentation mining models')
    parser.add_argument('model_name', type=str, default='', help='The abbreviature of the model to test: FtEn; FtPt; MSuEn; MUnEn; VMSuEn; VMUnEn; MSuPt; MUnPt; VMSuPt; VMUnPt; MSuEnPt; MUnEnPt; VMSuEnPt; VMUnEnPt')
    parser.add_argument('--cuda', type=bool_flag, default=False, help="Run on GPU")

    args = parser.parse_args()

    assert args.model_name in ["FtEn", "FtPt", "MSuEn", "MUnEn", "VMSuEn", "VMUnEn", "MSuPt", "MUnPt", "VMSuPt", "VMUnPt", "MSuEnPt", "MUnEnPt", "VMSuEnPt", "VMUnEnPt"]

    newDirectory = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    dumpPath = r'Dumps/' + newDirectory
    while os.path.exists(dumpPath):
        newDirectory = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        dumpPath = r'Dumps/' + newDirectory
    os.makedirs(dumpPath)

    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(42)
    rn.seed(42)
    tf.set_random_seed(42)

    if 'EnPt' in args.model_name:
        textDirectory = 'CorpusOutputPunctuation/txt/texts'
        tagDirectory = 'CorpusOutputPunctuation/txt/tags'
        addTexts = True
    elif 'Pt' in args.model_name:
        textDirectory = 'CorpusOutputPunctuation/txt/texts'
        tagDirectory = 'CorpusOutputPunctuation/txt/tags'
        addTexts = False
    else:
        textDirectory = 'essaysClaimsPremisesPunctuation/texts'
        tagDirectory = 'essaysClaimsPremisesPunctuation/tags'
        addTexts = False

    embeddings = args.model_name + 'Emb'

    if args.cuda:
        with tf.device('/gpu:0'):
            fullSequence(textDirectory, tagDirectory, addTexts, embeddings, dumpPath)
    else:
        with tf.device('/cpu:0'):
            fullSequence(textDirectory,tagDirectory, addTexts, embeddings, dumpPath)


if __name__ == '__main__':
    main()
