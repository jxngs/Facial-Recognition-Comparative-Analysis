
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout,QLineEdit,QFormLayout,QMessageBox, QGridLayout
from heapq import *
from DataLoader import *
from FacialRecognition import *

class MLApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('ML Algorithm Selector')
        self.setGeometry(100, 100, 600, 300)

        algolabel = QLabel('Select your ML algorithm:')
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['', 'Eigenfaces', 'Fisherfaces', 'LBPH', 'CNN'])
        self.algorithm_combo.currentIndexChanged.connect(self.update_hyperparameters)

        datalabel = QLabel('Select your Dataset:')
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['', 'LFW', 'TA Data', 'Yale Data', 'BFW Balanced', 'BFW Probabilistic'])
        self.dataset_combo.currentIndexChanged.connect(self.update_hyperparameters)
        self.hyperparam_layout = QFormLayout()

        self.num_features_widget = QLineEdit(str(150))
        self.train_test_split_widget = QLineEdit(str(0.8))
        self.additional_widget = QLineEdit("")
        self.num_features_label = QLabel("Number of Features:")
        self.train_test_split_label = QLabel("Train/Test Split Proportion:")
        self.additional_label = QLabel("")
        self.num_features_widget.setVisible(False)
        self.train_test_split_widget.setVisible(False)
        self.additional_widget.setVisible(False)
        self.num_features_label.setVisible(False)
        self.train_test_split_label.setVisible(False)
        self.additional_label.setVisible(False)
        self.hyperparam_layout.addRow(self.num_features_label, self.num_features_widget)
        self.hyperparam_layout.addRow(self.train_test_split_label, self.train_test_split_widget)
        self.hyperparam_layout.addRow(self.additional_label, self.additional_widget,)
        self.men_subset_label = QLabel("Male-Identifying %:")
        self.women_subset_label = QLabel("Female-Identifying %:")
        self.caucasian_subset_label = QLabel("Caucasian-Identifying %:")
        self.black_subset_label = QLabel("Black-Identifying %:")
        self.indian_subset_label = QLabel("Indian-Identifying %:")
        self.asian_subset_label = QLabel("Asian-Identifying %:")
        self.men_subset_widget = QLineEdit()
        self.women_subset_widget = QLineEdit()
        self.caucasian_subset_widget = QLineEdit()
        self.black_subset_widget = QLineEdit()
        self.indian_subset_widget = QLineEdit()
        self.asian_subset_widget = QLineEdit()
        self.demographic_widgets = [self.men_subset_label,self.women_subset_label,self.caucasian_subset_label,self.black_subset_label,self.asian_subset_label,self.indian_subset_label,self.men_subset_widget,self.women_subset_widget,self.caucasian_subset_widget,self.black_subset_widget, self.indian_subset_widget, self.asian_subset_widget]
        self.demographic_layout = QGridLayout()
        self.demographic_layout.addWidget(self.men_subset_label, 0, 0)
        self.demographic_layout.addWidget(self.men_subset_widget, 0, 1)
        self.demographic_layout.addWidget(self.women_subset_label, 0, 2)
        self.demographic_layout.addWidget(self.women_subset_widget, 0, 3)

        self.demographic_layout.addWidget(self.asian_subset_label, 2, 0)
        self.demographic_layout.addWidget(self.asian_subset_widget, 2, 1)
        self.demographic_layout.addWidget(self.black_subset_label, 2, 2)
        self.demographic_layout.addWidget(self.black_subset_widget, 2, 3)

        self.demographic_layout.addWidget(self.caucasian_subset_label, 3, 0)
        self.demographic_layout.addWidget(self.caucasian_subset_widget, 3, 1)
        self.demographic_layout.addWidget(self.indian_subset_label, 3, 2)
        self.demographic_layout.addWidget(self.indian_subset_widget, 3, 3)

        for widget in self.demographic_widgets:
            widget.setVisible(False)

        train_test_button = QPushButton('Train/Test Algorithm')
        train_test_button.clicked.connect(self.train_test_algorithm)

        layout = QVBoxLayout()
        layout.addWidget(algolabel)
        layout.addWidget(self.algorithm_combo)
        layout.addWidget(datalabel)
        layout.addWidget(self.dataset_combo)
        layout.addLayout(self.demographic_layout)
        layout.addLayout(self.hyperparam_layout)
        layout.addWidget(train_test_button)

        self.setLayout(layout)
        self.show()

    def update_hyperparameters(self):
        if self.algorithm_combo.currentText() == "":
            self.num_features_widget.setVisible(False)
            self.train_test_split_widget.setVisible(False)
            self.additional_widget.setVisible(False)
            self.num_features_label.setVisible(False)
            self.train_test_split_label.setVisible(False)
            self.additional_label.setVisible(False)
        else:
            self.num_features_widget.setVisible(True)
            self.train_test_split_widget.setVisible(True)
            self.additional_widget.setVisible(True)
            self.num_features_label.setVisible(True)
            self.train_test_split_label.setVisible(True)
            self.additional_label.setVisible(True)
            self.num_features_widget.setText(str(150))
            if self.algorithm_combo.currentText() == "Eigenfaces":
                self.additional_widget.setText(str(1.0))
                self.additional_label.setText("C Value:")
            if self.algorithm_combo.currentText() == "Fisherfaces":
                self.num_features_widget.setText(str(40))
                self.additional_widget.setVisible(False)
                self.additional_label.setVisible(False)
            if self.algorithm_combo.currentText() == "LBPH":
                self.additional_widget.setText("ChiSquare")
                self.additional_label.setText("Distance Metric:")
            if self.algorithm_combo.currentText() == "CNN":
                self.additional_widget.setText(str(75))
                self.additional_label.setText("Number of Epochs:")
        if self.dataset_combo.currentText()=="BFW Probabilistic":
            for widget in self.demographic_widgets:
                widget.setVisible(True)
        else:
            for widget in self.demographic_widgets:
                widget.setVisible(False)
        

    def validate_proportions(self):
        men_women_proportion = sum(float(widget.text()) for widget in self.demographic_widgets[6:8] if widget.text())
        others_proportion = sum(float(widget.text()) for widget in self.demographic_widgets[8:]if widget.text())
        if men_women_proportion != 100.0 :
            QMessageBox.warning(self, "Invalid Proportions", "Men and Women must add up to 100.")
            return False
        if others_proportion != 100.0:
            QMessageBox.warning(self, "Invalid Proportions", "Racial Groups must add up to 100.")
            return False
        
        Asian = float(self.demographic_widgets[11].text())
        Black = float(self.demographic_widgets[9].text())
        Indian = float(self.demographic_widgets[10].text())
        White = float(self.demographic_widgets[8].text())
        Male = float(self.demographic_widgets[6].text())
        Female = float(self.demographic_widgets[7].text())
        
        self.probabilities = {
            "asian_females": Asian * Female,
            "black_females": Black * Female,
            "indian_females": Indian * Female,
            "white_females": White * Female,
            "asian_males": Asian * Male,
            "black_males": Black * Male,
            "indian_males": Indian * Male,
            "white_males": White * Male
        }
        return True

    def train_test_algorithm(self):
        
        datasettype = self.dataset_combo.currentText()
        modeltype = self.algorithm_combo.currentText()
        num_features = int(self.num_features_widget.text())
        train_test_split = float(self.train_test_split_widget.text())
        self.probabilities={}
        if datasettype == "LFW":
            dataset = LFW()
        elif datasettype=="":
            return
        elif datasettype == "TA Data":
            dataset = TA_Data()
        elif datasettype == "Yale Data":
            dataset = Yale_Data()
        elif datasettype == "BFW Probabilistic":
            if not self.validate_proportions():
                return
            dataset = BFW_Probabilistic()
        elif datasettype == "BFW Balanced":
            dataset = BFW_Balanced()

        dataset.load_data(modeltype, num_features=num_features, prop=1 - train_test_split, probabilities = self.probabilities)

        if modeltype == 'Eigenfaces':
            print("Running Eigenfaces")
            c_value = float(self.additional_widget.text())
            f = Eigenfaces(modeltype, dataset, num_features,c_value=1)
        elif modeltype == 'Fisherfaces':
            print("Running Fisherfaces")
            f = FisherFaces(modeltype, dataset, num_features)
        elif modeltype == 'LBPH':
            print("Running LBPH")
            metric = self.additional_widget.text()
            f = LBPH(modeltype, dataset, num_features, metric)
        elif modeltype == 'CNN':
            print("RUNNING CNN")
            epochs = int(self.additional_widget.text())
            f = CNN(modeltype, dataset, epochs)

            # Print all hyperparameter values and the dataset being used on separate lines
        print(f"Number of Features: {num_features}")
        print(f"Train/Test Split Proportion: {train_test_split}")
        print(f"Dataset being used: {datasettype}")
        if modeltype == 'Eigenfaces':
            print(f"C Value: {c_value}")
        if modeltype == "LBPH":
            print(f"Distance Metric: {metric}")
        if modeltype == "CNN":
            print(f"# of Epochs: {epochs}")
        f.X_train, f.X_test, f.y_train, f.y_actual = f.dataset.X_train, f.dataset.X_test, f.dataset.y_train, f.dataset.y_actual
        f.train()
        f.predict()
        f.show_results()
if __name__ == '__main__':
 
    app = QApplication(sys.argv)
    ml_app = MLApp()
    ml_app.show()
    sys.exit(app.exec_())

