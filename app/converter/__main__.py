"""Entry point for python -m app.converter."""

import sys
from PySide6.QtWidgets import QApplication
from app.converter.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
