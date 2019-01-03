from model import VAE
import torch
from torch.nn import functional as F
from torchvision import datasets,transforms

# Hyper Parameters
image_size = 784
batch_size = 128
learning_rate = 1e-3
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,transform=transforms.ToTensor()),batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),batch_size=batch_size, shuffle=True)

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i, (x,_) in enumerate(train_loader):
        # Forward
        x = x.to(device).view(-1, image_size)
        x_const, mu, logvar = model(x)
        # Compute KL Divergence and reconstruction loss
        decoder_loss = F.binary_cross_entropy(x_const,x,size_average=False)
        KL_div = -0.5 * torch.sum(1+logvar-mu.pow(2)-logvar.exp()) # KL Divergence
        # BackPropogate
        loss = decoder_loss+KL_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
                  .format(epoch + 1, num_epochs, i + 1, len(train_loader), decoder_loss.item(), KL_div.item()))

torch.save(model.state_dict(),"trained_model.ckpt")