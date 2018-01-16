def main():
    training_data = [[1.00,	0.08,	0.72,	1.0],
                    [1.00,	0.10,	1.00,	0.0],
                    [1.00,	0.26,	0.58,	1.0],
                    [1.00,	0.35,	0.95,	0.0],
                    [1.00,	0.45,	0.15,	1.0],
                    [1.00,	0.60,	0.30,	1.0],
                    [1.00,	0.70,	0.65,	0.0],
                    [1.00,	0.92,	0.45,	0.0]]

    weights = [0.20, 1.00, -1.00]

    epochs = 10
    eta = 1.0
    train_weights(training_data, weights)


def plot(matrix,weights):
    pass


def predict(input, weights):
    activation = sum(list(map(lambda x: x[0]*x[1], list(zip(input, weights)))))
    return 1.0 if activation > 0.0 else 0.0


def accuracy(matrix, weights):
    correct_predictions = 0
    predictions = []
    for row in matrix:
        res = predict(row[:-1], weights)
        predictions.append(res)
        if res == row[-1]:
            correct_predictions += 1

    return correct_predictions/float(len(matrix))


def train_weights(matrix, weights, epochs=10, eta=1.0):
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        current_predicition = accuracy(matrix, weights)
        print("Current prediction {}".format(current_predicition))
        if current_predicition == 1.0:
            print("Training complete")
            break

        for row in matrix:
            prediction = predict(row[:-1], weights)
            error = row[-1]-prediction

            # Update weights
            for j in range(len(weights)):
                weights[j] += eta*error*row[j]

    print("Accuracy after training: {}".format(accuracy(matrix, weights)))
    return weights

if __name__ == '__main__':
    main()
