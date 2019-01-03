from model import VAE
import torch
from torchvision import datasets,transforms
from torchvision.utils import save_image

# Hyper Parameters
image_size = 784
batch_size = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = VAE()
model.eval()
model.load_state_dict(torch.load('trained_model.ckpt'))
model.to(device)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),batch_size=batch_size, shuffle=True)

# Generate one batch of testing sample
test_sample,_ = next(iter(test_loader))
test_sample = test_sample.to(device).view(-1,784)

with torch.no_grad():
    # Sampled Image from decoder
    z = torch.randn(batch_size, 20).to(device)
    sampled = model.decoder(z).view(-1, 1, 28, 28)
    save_image(sampled, 'sampled.png')

    # Reconstructed from test sample
    reconstructed,_,_ = model(test_sample)
    reconstructed = reconstructed.view(-1,1,28,28)
    test_sample = test_sample.view(-1,1,28,28)
    x_concat = torch.cat([reconstructed,test_sample],dim=2)
    print(x_concat.shape)
    save_image(x_concat, 'reconstructed.png')
