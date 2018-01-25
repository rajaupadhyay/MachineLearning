import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def main():
    training_data = [[1.00,	0.08,	0.72,	1.0],
                    [1.00,	0.10,	1.00,	0.0],
                    [1.00,	0.26,	0.58,	1.0],
                    [1.00,	0.35,	0.95,	0.0],
                    [1.00,	0.45,	0.15,	1.0],
                    [1.00,	0.60,	0.30,	1.0],
                    [1.00,	0.70,	0.65,	0.0],
                    [1.00,	0.92,	0.45,	0.0],
                    [1.00,  0.09,   1.20,   0.0]]

    training_data2 = [	[1.0,	0.02, 	0.48,	0.0],
                        [1.0,	0.08, 	0.72,	1.0],
                        [1.0,	0.10, 	1.00,	0.0],
                        [1.0,	0.20, 	0.50,	1.0],
                        [1.0,	0.24, 	0.30,	1.0],
                        [1.0,	0.35, 	0.35,	1.0],
                        [1.0,	0.36, 	0.75,	0.0],
                        [1.0,	0.45, 	0.50,	1.0],
                        [1.0,	0.52, 	0.24,	0.0],
                        [1.0,	0.70, 	0.65,	0.0],
                        [1.0,	0.80, 	0.26,	0.0],
                        [1.0,	0.92, 	0.45,	0.0]]

    weights = [0.20, 1.00, -1.00]

    epochs = 10
    eta = 1.0
    train_weights(training_data, weights)
    # train_weights(training_data2, weights)


def plot(matrix,weights):
    # Add line through plot
    x = [row[1] for row in matrix]
    y = [row[2] for row in matrix]
    cl = [row[-1] for row in matrix]
    print(x, y)
    indices_1 = [i for i in range(len(cl)) if cl[i] == 1.0]
    x_label1 = [x[i] for i in indices_1]
    y_label1 = [y[i] for i in indices_1]

    indices_2 = [i for i in range(len(cl)) if cl[i] == 0.0]
    x_label2 = [x[i] for i in indices_2]
    y_label2 = [y[i] for i in indices_2]

    plt.scatter(x_label1, y_label1, c='r', label="Class A")
    plt.scatter(x_label2, y_label2, c='b', label="Class B")
    plt.legend()

    # Plot Separator
    # line = [(weights[0]/weights[1]), (weights[0]/weights[2])]
    # print(line)
    plt.plot([0, -(weights[0]/weights[2])], [-(weights[0]/weights[1]), 0])

    plt.show()


def predict(input, weights):
    activation = sum(list(map(lambda x: x[0]*x[1], list(zip(input, weights)))))
    return 1.0 if activation > 0.0 else 0.0


def accuracy(matrix, weights):
    correct_predictions = 0.0
    predictions = []
    for row in matrix:
        res = predict(row[:-1], weights)
        predictions.append(res)
        if res == row[-1]:
            correct_predictions += 1.0

    return correct_predictions/float(len(matrix))


def train_weights(matrix, weights, epochs=10, eta=1.0):
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        current_accuracy = accuracy(matrix, weights)
        print("Current prediction {}".format(current_accuracy))
        if current_accuracy == 1.0:
            print("Training complete")
            break
        for row in matrix:
            prediction = predict(row[:-1], weights)
            print("PREDICTION VS REAL: {} {}".format(prediction, row[-1]))
            error = row[-1]-prediction

            # Update weights
            for j in range(len(weights)):
                weights[j] += eta*error*row[j]

        plot(matrix, weights)

    print("Accuracy after training: {}".format(accuracy(matrix, weights)))
    plot(matrix, weights)
    return weights

if __name__ == '__main__':
    main()
