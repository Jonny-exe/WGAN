from torchvision import datasets, transforms
from discriminator import Discriminator
from generator import Generator
from math import log
from helpers import show_generated_images, plot_gan_losses
import matplotlib.pyplot as plt

import torch
import sys
import numpy as np


BATCH_SIZE = 128
EPOCHS = 50
ITERATIONS = 100

transform = transforms.Compose(
    [transforms.ToTensor()]
# , transforms.Normalize((0.1307,), (0.3081,))]
)
print(sys.argv)

if len(sys.argv) > 1 and sys.argv[1] == "test":
    G = Generator()  # Create the model architecture

    G.load_state_dict(torch.load('G.pth'))  # Load saved weights

    G.eval()
    
    show_generated_images(G, 100)
    exit()


     


dataset1 = datasets.MNIST("./data", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST("./data", train=False, transform=transform)


train_loader = torch.utils.data.DataLoader(dataset1, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)


D = Discriminator(k=BATCH_SIZE)
G = Generator()


# Best results yet, 0.0001 and 0.001

# Good loss resulsts with 0.00008 and 0.008. See ./img/Figure_2.png. However discriminator told 0.2 fake, and 0.5 real. Not good enough discrimiantor. Ratio 1/4
# Tried with ratio 1/3. D overpowered G. Images come up empty
# Increased LR for G, to 0.012, to see if it doesn't get overpowered
# This didn't help. Loss didn't go down for G. Probably LR to big
# Dereased LR to see if got better, to 0.005
# Now G to strong. Images blob
# Decreased ratio to 1/2
# D WAY to strong. Way to strong
# Tried wth ratio 1/2 + 1/3.
# A bit better, but in the end D still too strong
# Decreased G Lr, to 0.002 even more, maybe better resulsts
# Decreased G Lr, to 0.0005, and went back to 1/3 ratio
# Best yet. Graphs look ok. But images is only circles in the middle. D a bit to low.
# To see if D goes faster added back the first head start
# Now D to strong. 
# Decreased LR even more for D, to 0.00004
# Went back to 0.00008 for D LR. Added dropout to D
# D to strong
# Went back to 0.00006 for D LR.
# D still to strong
# Went back to 0.0008 for G LR.
# Best resulsts yet. WIth ratio 2/5, 0.00006 and 0.005

optim_d = torch.optim.Adam(D.parameters(), lr=0.0000075, betas=(0.5, 0.999))
optim_g = torch.optim.Adam(G.parameters(), lr=0.00025, betas=(0.5, 0.999))


D.to("cuda")
G.to("cuda")

loss =  0
g_losses = [0]
d_losses = [0]

fig, ax = plt.subplots(figsize=(10, 5))
plt.ion()    # <-- TURN ON INTERACTIVE MODE (Important!)
plt.show() 

for e in range(1, EPOCHS):
    print("EPOCH: ", e)
    i = 0
    if e % 5 == 0:
        show_generated_images(G, 100)
    for train_features, train_labels in iter(train_loader):
        # print(train_features.size())
        # print(train_labels.size(), train_labels)
        train_features = train_features.to("cuda")
        train_labels = train_labels.to("cuda")

        optim_d.zero_grad()
        optim_g.zero_grad()



        z = torch.randn(BATCH_SIZE, 100) # TODO this should be gaussian 
        z = z.to("cuda")

        loss = 0

        # if (e % 2 == 1 or e <= 2) 
        # if (e % 6 == 1) or (e % 6 == 3):
        # print(i % 5 <= 1)
        # if (e < 10 and i % 4 == 2) or (e % 3 == 0 and i > 10):
        # if (e < 10 and i % 4 == 3) or (e >= 10 and i % 3 == 0):
        # print(g_losses[-1], d_losses[-1])
        # if i % 100 == 0:
        #     print(g_losses[-1], d_losses[-1])
        # if (g_losses[-1] < d_losses[-1]) or (e <= 1 and i % 5):
        if i % 5 <= 3:
            fake_image = G.forward(z).detach()
            validity_fake = D.forward(fake_image)
            train_features += (torch.randn(BATCH_SIZE, 1, 28, 28) * 0.1).to("cuda")

            validity_real = D.forward(train_features)
            #if i % 10 == 0:
                #print("Loss fake: ", torch.mean(validity_fake).item(), torch.mean(validity_real).item())
                # print("Loss real: ", 

            # loss_fake = -torch.log(validity_fake)
            # loss_real = torch.log(validity_real)

            loss = -torch.mean(validity_real - validity_fake)

            if i % 10 == 0:
                print("Loss: ", loss.item())
            loss.backward()
            optim_d.step()


            with torch.no_grad():
                for param in D.parameters():
                    param.clamp_(-0.01, 0.01)

            d_losses.append(loss.item())
        else:
            fake_image = G.forward(z)
            validity = D.forward(fake_image)

            #if i % 25 == 0:
                #print("Generated: ", torch.mean(validity).item())

            # print("out_g: ", out_g)
            # print(torch.round(out_g))

            # loss = torch.log(1 - validity) # Saturating
            # loss = -torch.log(validity)
            loss = torch.mean(-validity)
            loss.backward()
            optim_g.step()

            g_losses.append(loss.item())
        i += 1
    print(i)
    print(np.mean(d_losses[:-10]), np.mean(g_losses[:-10]))
        


    plot_gan_losses(g_losses, d_losses, ax, fig)

print("Finished")


show_generated_images(G, 100)

torch.save(D.state_dict(), 'D.pth')
torch.save(G.state_dict(), 'G.pth')



