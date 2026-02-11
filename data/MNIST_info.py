from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = "../data"
bs = 128

tfm = transforms.ToTensor()
mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)

print("len(mnist_train) =", len(mnist_train))

idx0 = (mnist_train.targets == 0).nonzero(as_tuple=True)[0].tolist()
ds0 = Subset(mnist_train, idx0)
loader0 = DataLoader(ds0, batch_size=bs, shuffle=True)

print("len(ds0) =", len(ds0))
print("len(loader0) =", len(loader0))
x, y = next(iter(loader0))
print("batch y unique =", y.unique())
print(x.shape, y.shape)
