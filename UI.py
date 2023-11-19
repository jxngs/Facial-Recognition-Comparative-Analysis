import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QComboBox, QPushButton, QVBoxLayout
from cnn import CNNRunner
from fisher_driver import *
from eigenfaces import *
from LBPHdriver import lfw_test

class MLApp(QWidget):
    def __init__(self):
     
        super().__init__()

        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('ML Algorithm Selector')
        
        self.setGeometry(100, 100, 600, 300)  # Adjusted the window size


        # Create widgets
        label = QLabel('Select your ML algorithm:')
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['','Eigenfaces', 'Fisherfaces', 'LBPH', 'CNN'])
        train_test_button = QPushButton('Train/Test Algorithm')
        train_test_button.clicked.connect(self.train_test_algorithm)
        # Set up the layout
        layout = QVBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.algorithm_combo)
        layout.addWidget(train_test_button)
        self.setLayout(layout)
        self.show()

    def train_test_algorithm(self):
        selected_algorithm = self.algorithm_combo.currentText()
    
        if selected_algorithm == 'Eigenfaces':
            print("TRAINING")
            eigenrunner()     
        elif selected_algorithm == 'Fisherfaces':
            print("TRAINING")
            fisher_runner(40)
        elif selected_algorithm == 'LBPH':
            print("TRAINING")
            lfw_test(50)
        elif selected_algorithm == 'CNN':
            print("TRAINING")
            CNNRunner()
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ml_app = MLApp()
    ml_app.show()
    sys.exit(app.exec_())
