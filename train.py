import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from tqdm import tqdm
from dataloader import FacadeDataset, ToTensor

from model import Generator, Discriminator

MAX_EPOCHS = 500
LAMBDA = 1
BATCH_SIZE = 4

transform=transforms.Compose([ToTensor()])

train_set = FacadeDataset("datasets/facades/train", transform=transform)
val_set = FacadeDataset("datasets/facades/val", transform=transform)
test_set = FacadeDataset("datasets/facades/test", transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, \
                    shuffle=True, num_workers=0)
generator = Generator().cuda()
discriminator = Discriminator().cuda()
gen_optim = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
dis_optim = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

gen_loss = torch.tensor(0)
dis_loss = torch.tensor(0)

for epoch in range(MAX_EPOCHS):
    pbar = tqdm(train_loader, dynamic_ncols=True)
    pbar.set_description(f'e{epoch}')
    for idx, batch in enumerate(pbar):

        image0, image1 = batch["image0"], batch["image1"]
        image0, image1 = image0.cuda(), image1.cuda()

        image1_pred = generator(image0)
        d_real = discriminator(image0, image1).mean(dim=0)
        d_fake = discriminator(image0, image1_pred).mean(dim=0)
        l1_loss = torch.nn.functional.l1_loss(image1, image1_pred)

        if idx % 2 == 0:
            gen_loss = -d_fake + LAMBDA * l1_loss
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()
        else:
            dis_loss = d_real + torch.log(1 - torch.exp(d_fake))
            dis_loss /= 2
            dis_optim.zero_grad()
            dis_loss.backward()
            dis_optim.step()

        pbar.set_description(f'gloss={gen_loss.item()}; dloss={dis_loss.item()}')

