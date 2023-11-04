from NN import *
from Data import *


def test_for_nn():
    amount = 100
    # function: y = x^2
    features = np.linspace(-1, 1, amount).reshape((amount, 1))
    labels = np.power(features, 2)

    plt.figure(figsize=(5, 5), dpi=150)
    plt.scatter(features[:, 0], labels[:, 0])

    u = MyNet()
    pred_u = u(features)
    plt.plot(features, pred_u, c='r')
    plt.show()


def test_for_data():
    gor1 = Generator(300, 0, method='uniform')
    gor2 = Generator(500, 1, method='uniform')
    data1 = Process.join(
        gor1.line((-1, -1), (1, 1), 0.2),
        gor1.line((1, -1), (-1, 1), 0.2)
    )
    data2 = gor2.circle_inside((0, 0), 0.5)
    data = Process.join(data1, data2)

    plot = Plot2d(figsize=(5, 5), title='Test', save=True)
    plot(data)


if __name__ == '__main__':
    test_for_data()





