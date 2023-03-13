X = 2
Y = 4


W = 0.0


def forward(x,w):
    return w*x


def loss(y,y_pred):
    return 1/1 * ((y_pred-y)**2)


# MSE
# 1/N * (y_pred - y)**2
# 1/N * 2*(y_pred-y) * x
# 1/N * 2*x*(y_pred-y)
def gradient(x,y,y_pred):
    return 1/1 *2*x*(y_pred-y)



num_iter = 90
lr = 0.01

for epoch in range(num_iter):

    y_pred = forward(X,W)

    l = loss(Y,y_pred)

    grad = gradient(X,Y,y_pred)

    # "-" because in gradient calculation we do y_pred-y
    # could be "+" if we did y-y_pred
    W -= grad*lr

    print(f"E: {epoch} , W: {W:.2f}, L: {l:.2f}, G: {grad:.2f} P: {forward(X,W):.2f} ")
    print("-----------------------------------------------")