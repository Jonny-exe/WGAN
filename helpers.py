import torch
import matplotlib.pyplot as plt

def show_generated_images(generator, noise_dim, epochs, device='cpu', num_images=16):
    generator.eval()  # Set to eval mode
    generator.to(device)
    plt.ioff()

    with torch.no_grad():
        # z = torch.randn(num_images, noise_dim, device=device)
        z = torch.randn(16, 100, device=device)
        fake_images = generator(z).cpu()

    # Reshape and scale for display
    fake_images = fake_images.view(num_images, 1, 28, 28)
    # fake_images = (fake_images + 1) / 2  # If using Tanh, convert to [0,1]
    # fake_images = fake_images * 255

    grid_size = int(num_images**0.5)
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(fake_images[i][0], cmap='gray', vmin=-1, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('out_' + "0" * (3-len(str(epochs)))+ str(epochs) + '.png')
    plt.savefig('out.png')

    generator.train()  # Set to eval mode
    generator.to("cuda")


# first = True
# def plot_gan_losses(g_losses, d_losses, epoch):
#     if epoch == 1:
#         plt.figure(figsize=(10,5))
#         first = False
#     else:
#         plt.clf()  # Clear the figure
#     plt.title("Generator and Discriminator Loss During Training")
#     plt.plot(g_losses, label="G Loss")
#     plt.plot(d_losses, label="D Loss")
#     plt.xlabel("Iterations")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.show(block=False)
#     plt.pause(0.001)



def plot_gan_losses(g_losses, d_losses, ax, fig):
    # print(g_losses, d_losses)
    ax.clear()  # Clear only the axes, keep figure
    ax.set_title("Generator and Discriminator Loss During Training")  # <-- ax not fig!
    ax.plot(g_losses, label="G Loss")
    ax.plot(d_losses, label="D Loss")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Loss")
    ax.legend()
    print("Draw ")
    fig.canvas.draw()  # Explicitly ask the figure to redraw
    fig.canvas.flush_events()  # Force GUI to update
    fig.savefig('graph.png')

