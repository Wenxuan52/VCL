from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader

DATA_DIR = "../data"

tfm = transforms.Compose([transforms.ToTensor()])

mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
mnist_test  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)

def make_digit_subset(ds, digit: int):
    targets = ds.targets  # tensor of labels
    idx = (targets == digit).nonzero(as_tuple=True)[0].tolist()
    return Subset(ds, idx)

task_train_sets = [make_digit_subset(mnist_train, d) for d in range(10)]
task_test_sets  = [make_digit_subset(mnist_test,  d) for d in range(10)]  # 可选

# 举例：task 0 的 loader
train_loader_0 = DataLoader(task_train_sets[0], batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

print(len(train_loader_0))          # 60000
x, y = next(iter(train_loader_0))
print(x.shape, y.shape)        # [B,1,28,28] [B]
