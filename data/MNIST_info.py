from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

DATA_DIR = "../data"
bs = 128

tfm = transforms.ToTensor()
mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)

print("len(mnist_train) =", len(mnist_train))  # 应该是 60000

# 只取 digit=0
idx0 = (mnist_train.targets == 0).nonzero(as_tuple=True)[0].tolist()
ds0 = Subset(mnist_train, idx0)
loader0 = DataLoader(ds0, batch_size=bs, shuffle=True)

print("len(ds0) =", len(ds0))          # 大约 5923
print("len(loader0) =", len(loader0))  # 47（或接近这个数）
x, y = next(iter(loader0))
print("batch y unique =", y.unique())
print(x.shape, y.shape)
