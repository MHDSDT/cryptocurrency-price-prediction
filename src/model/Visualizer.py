from matplotlib import pyplot as plt


class Visualizer:
    @staticmethod
    def plot_two_lines(line1, line2, label1=None, label2=None, lw=2):
        fig, ax = plt.subplots(1, figsize=(13, 7))
        ax.plot(line1, label=label1, linewidth=lw)
        ax.plot(line2, label=label2, linewidth=lw)
        ax.legend(loc='best', fontsize=16)
        plt.show()

    @staticmethod
    def plot_train_val_loss(history):
        plt.plot(history.history['loss'], 'r', linewidth=2, label='Train loss')
        plt.plot(history.history['val_loss'], 'g', linewidth=2, label='Validation loss')
        plt.title('LSTM')
        plt.xlabel('Epochs')
        plt.ylabel('MSE')
        plt.show()
