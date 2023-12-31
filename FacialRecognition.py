

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import keras
from tensorflow.keras import layers
from PIL import Image
class FacialRecognitionAlgorithm:
    def __init__(self, modelname, dataset):
        self.modelname = modelname
        self.dataset=dataset

    def train(self):
        pass

    def predict(self):
        pass

    def show_results(self):
        pass

    def print_cross_sectional(self, model='', show=True):

        if not hasattr(self.dataset, 'labels_to_demographics'): return

        self.accuracy = accuracy_score(self.y_actual, self.y_predicted)
        self.demographic_accuracy = {}
        self.gender_accuracy = {}
        self.race_accuracy = {}
        # split self.y_actual into groups
        for i, person in enumerate(self.y_actual):

            demo_group = self.dataset.labels_to_demographics[person]
            race, gender = demo_group.split('_')

            self.demographic_accuracy.setdefault(demo_group, [])
            self.race_accuracy.setdefault(race, [])
            self.gender_accuracy.setdefault(gender, [])

            num = 0
            if person == self.y_predicted[i]: num = 1
            self.demographic_accuracy[demo_group].append(num)
            self.race_accuracy[race].append(num)
            self.gender_accuracy[gender].append(num)

        # race and gender
        for demographic in self.race_accuracy.keys():
                res = self.race_accuracy[demographic]
                if len(res) > 0:
                    print(demographic, sum(res)/len(res))

        for demographic in self.gender_accuracy.keys():
                res = self.gender_accuracy[demographic]
                if len(res) > 0:
                    print(demographic, sum(res)/len(res))

        if show: 
            plot_acc = {}
            for demographic in self.demographic_accuracy.keys():
                res = self.demographic_accuracy[demographic]
                if len(res) > 0:
                    plot_acc[demographic] = sum(res)/len(res)
                    print(demographic, sum(res)/len(res))
        
            if model == 'figure': self.show_accuracy_hist_figure(plot_acc)    
            else: self.show_accuracy_hist(plot_acc)
            

    def show_accuracy_hist(self, plot_acc):
        
        # data to plot
        
        plt.subplot(1, 2, 1)

        sorted_dict = {k: plot_acc[k] for k in sorted(plot_acc.keys())}

        n_groups = 4
        female_acc = [value for key, value in sorted_dict.items() if 'females' in key]
        male_acc = [value for key, value in sorted_dict.items() if '_males' in key]

        # create plot
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, female_acc, bar_width,
        alpha=opacity,
        color='g',
        label='Females')

        rects2 = plt.bar(index + bar_width, male_acc, bar_width,
        alpha=opacity,
        color='r',
        label='Males')

        groups = sorted(self.race_accuracy.keys())

        plt.xlabel('Person')
        plt.ylabel('Scores')
        plt.title('Scores by person')
        plt.xticks(index + bar_width, groups)
        plt.legend()

        plt.subplot(1, 2, 2)
        
        
    def show_accuracy_hist_figure(self, plot_acc):
        
        sorted_dict = {k: plot_acc[k] for k in sorted(plot_acc.keys())}

        n_groups = 4
        female_acc = [value for key, value in sorted_dict.items() if 'females' in key]
        male_acc = [value for key, value in sorted_dict.items() if '_males' in key]

        # create plot
        index = np.arange(n_groups)
        bar_width = 0.35
        opacity = 0.8

        rects1 = plt.bar(index, female_acc, bar_width,
        alpha=opacity,
        color='g',
        label='Females')

        rects2 = plt.bar(index + bar_width, male_acc, bar_width,
        alpha=opacity,
        color='r',
        label='Males')

        groups = sorted(self.race_accuracy.keys())

        plt.xlabel('Person')
        plt.ylabel('Scores')
        plt.title('Scores by person')
        plt.xticks(index + bar_width, groups)
        plt.legend()

        plt.show()
        



class Eigenfaces(FacialRecognitionAlgorithm):

    class PrincipalComponentsAnalysis:
        def __init__(self, n_components):
            self.n_components = n_components
            self.mean = None
            self.components = None
            self.explained_variances = None

        def fit(self, X):
            self.mean = np.mean(X, axis=0)
            centered_data = X - self.mean
            covariance_matrix = np.cov(centered_data, rowvar=False)
            eigendata = np.linalg.eigh(covariance_matrix)
            eigenvalues = eigendata[0]
            eigenvectors = eigendata[1]
            sorted_indices = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, sorted_indices]
            self.components = eigenvectors[:, :self.n_components]
            total_variance = np.sum(eigenvalues)
            self.explained_variances = eigenvalues[sorted_indices][:self.n_components] / total_variance

        def transform(self, X):
            centered_data = X - self.mean
            projected_data = np.dot(centered_data, self.components)
            whitened_data = projected_data / np.sqrt(self.explained_variances)
            return whitened_data


    def __init__(self, modelname, dataset, num_features,c_value):
        self.n_components=num_features
        self.C=c_value
        super().__init__(modelname, dataset)
        
    def train(self):
        self.X_train = self.X_train.reshape((len(self.X_train), -1))
        self.X_test = self.X_test.reshape((len(self.X_test), -1))

        custom_pca = self.PrincipalComponentsAnalysis(n_components=self.n_components)
        custom_pca.fit(self.X_train)

        self.X_training_reduced = custom_pca.transform(self.X_train)
        self.X_test_reduced = custom_pca.transform(self.X_test)

    def predict(self):
        self.classifier = SVC(kernel='linear', C=self.C)
        self.classifier.fit(self.X_training_reduced, self.y_train)
        self.y_predicted = self.classifier.predict(self.X_test_reduced)
        self.accuracy = accuracy_score(self.y_actual, self.y_predicted)
        print(self.y_actual)

    def show_results(self):
        print(self.accuracy)
        self.print_cross_sectional('figure')
        # plt.subplot(1, 2, 2)

        ConfusionMatrixDisplay.from_estimator(
            self.classifier, self.X_test_reduced, self.y_actual, xticks_rotation="vertical"
        )
        plt.tight_layout()
        plt.show()

class FisherFaces(FacialRecognitionAlgorithm):
    def __init__(self, modelname, dataset, num_features):
        super().__init__(modelname, dataset)
        self.num_features = min(num_features, min(dataset.images[0].shape))

    def normalize(self, matrix, mean_all):
        min_val = np.min(mean_all)
        max_val = np.max(mean_all)
        normalized_matrix = 255 * (matrix - min_val) / (max_val - min_val)
        return normalized_matrix

    def train(self):
        
        n = len(self.X_train)
        unique_labels = list(set(self.y_train))
        self.classes = unique_labels
        class_means = []

        for i in range(len(unique_labels)):
            label = unique_labels[i]
            matching_indexes = np.where(self.y_train == label)[0]
            class_samples = self.X_train[matching_indexes]  # each c
            class_mean = np.mean(class_samples, axis=0)
            class_means.append(class_mean)
            
        class_means = np.array(class_means)
        mean_all = np.mean(class_means, axis=0)

        X = np.array([image for image in self.X_train])

        Sw = np.zeros((self.num_features, self.num_features))  # within class
        Sb = np.zeros((self.num_features, self.num_features))  # between-class

        for i in range(len(unique_labels)):
            label = unique_labels[i]
            class_samples = X[self.y_train == label]  # each c

            for class_sample in class_samples:
                Sw += np.dot((class_sample - class_means[i]).T, (class_sample - class_means[i]))

            Sb += len(class_samples) * np.dot((class_means[i] - mean_all), (class_means[i] - mean_all).T)

        eigenvalues, eigenvectors = np.linalg.eig(np.dot(np.linalg.inv(Sw), (Sb)))
        sort = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

        class_means = np.dot(class_means, eigenvectors)

        self.fisherfaces = eigenvectors
        self.mean_face = mean_all
        self.class_means = class_means
        self.basis = eigenvalues


    def predict(self):
        def predict1(image):
            projection = np.dot(image, self.fisherfaces)

            distances = []
            for c in self.class_means:
                distances.append(np.linalg.norm(projection - c))
            distances = np.array(distances)

            predicted_label = np.argmin(distances)

            distance = distances[predicted_label]
            return self.classes[predicted_label], distance
        self.accuracy = 0
        self.conf_matrix = np.zeros((len(self.dataset.numbers_to_labels), len(self.dataset.numbers_to_labels)))
        self.y_predicted=[]
        for x,y in zip(self.X_test, self.y_actual):
            
            pred = predict1(x)
            try:
                self.conf_matrix[self.dataset.labels_to_numbers[y], self.dataset.labels_to_numbers[pred[0]]] += 1
            except:
                self.conf_matrix[y, pred[0]] += 1
            self.y_predicted.append(pred[0])
            if y == pred[0]: self.accuracy += 1
        
        print('accuracy', self.accuracy/len(self.X_test))
        
    def show_results(self):
        self.print_cross_sectional()
        conf_matrix = confusion_matrix(self.y_actual, np.asarray(self.y_predicted), labels=self.classes)
        
        plt.imshow(conf_matrix, interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j],  'd'),
                        ha="center", va="center",
                        color="black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()


class CNN(FacialRecognitionAlgorithm):
    def __init__(self, modelname, dataset, epochs = 100):
        super().__init__(modelname, dataset)
        self.epochs=epochs
        
    def train(self):
    
        #train_images, val_images, train_labels, val_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            horizontal_flip=True,
        )

        datagen.fit(self.X_train)

        self.train_data_generator = datagen.flow(self.X_train, self.y_train, batch_size=64, shuffle=True)
        self.test_data_generator = datagen.flow(self.X_test, self.y_actual, batch_size=64, shuffle=True)

        # display first image in val data batch and its label -- just for validation
        first_batch = next(self.test_data_generator)
        #display(keras.preprocessing.image.array_to_img(first_batch[0][0] * 255))
        print(first_batch[1][0])

        self.test_data_generator = datagen.flow(self.X_test, self.y_actual, batch_size=64, shuffle=True) # reset val_data_generator

        # Xception model from https://keras.io/examples/vision/image_classification_from_scratch/#using-image-data-augmentation

        # def make_model(input_shape, num_classes):
        #     inputs = keras.Input(shape=input_shape)

        #     # Entry block
        #     x = layers.Rescaling(1.0 / 255)(inputs)
        #     x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
        #     x = layers.BatchNormalization()(x)
        #     x = layers.Activation("relu")(x)

        #     previous_block_activation = x  # Set aside residual

        #     for size in [256, 512, 728]:
        #         x = layers.Activation("relu")(x)
        #         x = layers.SeparableConv2D(size, 3, padding="same")(x)
        #         x = layers.BatchNormalization()(x)

        #         x = layers.Activation("relu")(x)
        #         x = layers.SeparableConv2D(size, 3, padding="same")(x)
        #         x = layers.BatchNormalization()(x)

        #         x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        #         # Project residual
        #         residual = layers.Conv2D(size, 1, strides=2, padding="same")(
        #             previous_block_activation
        #         )
        #         x = layers.add([x, residual])  # Add back residual
        #         previous_block_activation = x  # Set aside next residual

        #     x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        #     x = layers.BatchNormalization()(x)
        #     x = layers.Activation("relu")(x)

        #     x = layers.GlobalAveragePooling2D()(x)
        #     if num_classes == 2:
        #         activation = "sigmoid"
        #         units = 1
        #     else:
        #         activation = "softmax"
        #         units = num_classes

        #     x = layers.Dropout(0.5)(x)
        #     outputs = layers.Dense(units, activation=activation)(x)
        #     return keras.Model(inputs, outputs)


        # model = make_model(input_shape=IMAGE_SIZE + (1, ), num_classes=len(frequencies))

        input_shape = self.dataset.images[0].shape + (1, )
        self.num_classes = len(set(self.dataset.labels_to_numbers))

        self.model = keras.Sequential(
            [
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        next(self.train_data_generator)[1].shape
        self.callbacks = [
            keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
        ]
        self.model.compile(
            optimizer=keras.optimizers.legacy.Adam(1e-3), # legacy for m1 support
            loss="categorical_crossentropy",
                metrics=["accuracy"],
    )
        

    def predict(self):
        self.model.fit(
            self.train_data_generator,
            epochs=self.epochs,
            callbacks=self.callbacks,
            validation_data=self.test_data_generator,
        )

    
    def show_results(self):
        y_predicted = self.model.predict(self.X_test)
        
        # Convert predictions to class labels
        y_predicted = y_predicted.argmax(axis=1)
        y_predicted = np.asarray([self.dataset.numbers_to_labels[y_pred] for y_pred in y_predicted])
        self.y_predicted = y_predicted
        
        self.y_actual = [self.dataset.numbers_to_labels[np.where(row == 1)[0][0]] for row in self.y_actual]
        

        self.accuracy = accuracy_score(self.y_actual, y_predicted)
        print("Final Validation Accuracy: {:.2%}".format(self.accuracy))
        
        self.print_cross_sectional('figure')
        # Display confusion matrix
        cm = confusion_matrix(self.y_actual, y_predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(self.dataset.labels_to_numbers.keys(), key=lambda x: self.dataset.labels_to_numbers[x]))
        disp.plot(cmap='viridis', values_format='d', xticks_rotation="vertical")
        plt.tight_layout()
        plt.show()

     
class LBPH(FacialRecognitionAlgorithm):
    def __init__(self, modelname, dataset,num_features, metric, radius=1):
        super().__init__(modelname,dataset)
        self.size = num_features
        self.metric = metric
        self.radius=radius

    def train(self):
        self.classes = list(set(self.y_train))
        histograms = {}
        for i, image in enumerate(self.X_train):
            histograms.setdefault(self.y_train[i], [])
            histograms[self.y_train[i]].append(self.get_Histogram(image))
        self.histograms = histograms
    
    def get_Histogram(self, grayscaled_mat): 
        lbp_mat = self.get_LBP_Mat(grayscaled_mat)
        #plt.imsave('altered_images/lbp'+name+".jpeg", lbp_mat)
        lbp_values = lbp_mat.flatten()
        histogram = np.histogram(lbp_values, bins=256, range=(0, 256))
        # if show_hist:
        #     self.show_histogram(histogram, name)
        return histogram[0]
    
    def get_LBP_Mat(self, grayscaled_mat):
        M,N=grayscaled_mat.shape
        lbp_matrix = np.zeros((M, N), dtype=np.uint8)
        for r in range(1,M-1):
            for c in range(1,N-1):
                pixel_threshold = grayscaled_mat[r][c]
               # neighbors = self.getNeighbors(grayscaled_mat,r,c)
                neighbors = self.getRNeighbors(grayscaled_mat, r, c)
                above_thresh = neighbors >= pixel_threshold
                lbp_rc = np.sum(above_thresh * (2 ** np.arange(len(neighbors))))
                lbp_matrix[r, c] = lbp_rc
        return lbp_matrix


    def distance(self,vector1, vector2, metric):
        
        if metric == "ChiSquare":
            distance = np.sum((vector1 - vector2)**2 / (vector1 + vector2 + 1e-10))

        elif metric == "EuclideanDistance":
            distance = np.linalg.norm(vector1 - vector2)

        elif metric == "NormalizedEuclideanDistance":
            distance = np.linalg.norm(vector1 - vector2) / np.linalg.norm(vector1)

        elif metric == "AbsoluteValue":
            distance = np.sum(np.abs(vector1 - vector2))

        else:
            raise ValueError("Invalid metric. Supported metrics: 'ChiSquare', 'EuclideanDistance', 'NormalizedEuclideanDistance', 'AbsoluteValue'.")

        return distance

    def getRNeighbors(self, mat, r, c):
        arr = []
        for i in range(r - self.radius, r + self.radius + 1):
            if i not in range(self.size): continue
            for j in range(c - self.radius, c + self.radius + 1):
                if j not in range(self.size): continue
                arr.append(mat[i][j])
        return np.array(arr)

    def predict(self):
        def predict1(image):
            
            myhist=self.get_Histogram(image)
            distances = []
            ppl=[]
            for person in self.histograms:
                for histogram in self.histograms[person]:
                    distances.append(self.distance(histogram,myhist,self.metric))
                    ppl.append(person)
            distances = np.array(distances)

            ind = np.argmin(distances)

            pred_label, distance= ppl[ind], distances[ind]

            return pred_label, distance
        self.accuracy = 0
        self.conf_matrix = np.zeros((len(self.dataset.numbers_to_labels), len(self.dataset.numbers_to_labels)))
        self.y_predicted=[]
        for x,y in zip(self.X_test, self.y_actual):
            
            pred = predict1(x)
            try:
                self.conf_matrix[self.dataset.labels_to_numbers[y], self.dataset.labels_to_numbers[pred[0]]] += 1
            except:
                self.conf_matrix[y, pred[0]] += 1
            self.y_predicted.append(pred[0])
            if y == pred[0]: self.accuracy += 1
    def show_results(self):
        print('accuracy', self.accuracy/len(self.X_test))
        self.print_cross_sectional()
        conf_matrix = confusion_matrix(self.y_actual, np.asarray(self.y_predicted), labels=self.classes)
        
        plt.imshow(conf_matrix, interpolation='nearest')
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=45)
        plt.yticks(tick_marks, self.classes)

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j],  'd'),
                        ha="center", va="center",
                        color="black")

        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.show()

    
