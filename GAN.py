import utils
import torch
import torchvision
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import sys
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import PIL.Image, PIL.ImageTk


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Loads a given generator and generates a batch of images
# Images saved into GAN_result.jpg
def demonstrate_GAN(generator_name):
    generator = utils.Generator(100,64,3)
    generator.load_state_dict(torch.load('models/' + generator_name))
    noise = torch.randn(4, 100, 1, 1) #Generate batch of noise for input
    result = generator(noise).detach()
    result_grid = np.transpose(vutils.make_grid(result[:4], nrow=2, padding=1, normalize=True).cpu(),(1,2,0))
    plt.imsave('temp/GAN_demo.jpg', result_grid)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Trains a Generational Adversarial Network on a source image
# The Generator weights are saved in the trained_models folder
def train_GAN(source_image, learning_rate, iterations, generator_name, resume=False, model_name=None ):
    print(generator_name)
    device = torch.device('cuda') #Can change to cpu if you dont have CUDA
    criterion = nn.BCELoss()

    #Initialise networks and optimisers
    discriminator = utils.Discriminator128(64,3).to(device).apply(utils.initialise_weights)
    generator = utils.Generator128(100,64,3).to(device).apply(utils.initialise_weights)
    optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

    if resume:
        discriminator.load_state_dict(torch.load('models/' + model_name + '_disc.pt'))
        generator.load_state_dict(torch.load('models/' + model_name + '_gen.pt'))
        optimizerD.load_state_dict(torch.load('models/' + model_name + '_disc_optimiser.pt')) 
        optimizerG.load_state_dict(torch.load('models/' + model_name + '_gen_optimiser.pt')) 


    #Create dataset and dataloader from source image
    batch_size = 64
    dataset_size = 16384
    dataset = utils.TextureDataset(
        image_size=128,
        size=dataset_size,
        image_path=source_image,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    toPILImage = transforms.ToPILImage()

    count = 0
    for i in range(iterations):
        for j, data in enumerate(dataloader, 0):
            
            sys.stdout.write('\rIteration %d / %d, Image Batch %d / %d' % (i+1, iterations, j+1, len(dataloader)))
            sys.stdout.flush()
            data = data.to(device)    
            
            ## Train with all-real batch
            discriminator.zero_grad() 
            output = discriminator(data).view(-1) #Run real data through discriminator
            label = torch.full((batch_size,), 1, device=device) #Target is all 1's
            real_discriminator_error = criterion(output, label) #Get error from result and target
            real_discriminator_error.backward() #Backpropogate

            ## Train with all-fake batch
            noise = torch.randn(batch_size, 100, 1, 1, device=device) #Generate batch of noise for input
            fake = generator(noise) #Generate fake images from noise with generator
            label = torch.full((batch_size,), 0, device=device) #Target is all 0's
            output = discriminator(fake.detach()).view(-1) #Run fakes through discriminator
            fake_discriminator_error = criterion(output, label) #Get error from result and target
            fake_discriminator_error.backward() #Backpropogate

            optimizerD.step()

            # Train generator
            generator.zero_grad()
            label = torch.full((batch_size,), 1, device=device) #Target is all 1's
            output = discriminator(fake).view(-1) #Run fakes through discriminator again
            generator_error = criterion(output, label) #Get error from result and target
            generator_error.backward()#Backpropogate
            optimizerG.step()

            if j % 16 == 0: #Run noise through generator to print current results
                with torch.no_grad():
                    count = count + 1
                    current_images = generator(noise[:4]).detach().cpu() #Get image batch
                    current_images = toPILImage(vutils.make_grid(current_images[:4], nrow=2, padding=1, normalize=True).cpu())
                    current_images.save('temp/' +  generator_name  + str(count) + '.jpg')
                    f=open("temp/GANs.txt", "a+")
                    f.write("%d" % (0))
                    f.close()

    torch.save(generator.state_dict(), 'models/' + generator_name + '_gen.pt')
    torch.save(discriminator.state_dict(), 'models/' + generator_name + '_disc.pt')
    torch.save(optimizerG.state_dict(), 'models/' + generator_name + '_gen_optimiser.pt')
    torch.save(optimizerD.state_dict(), 'models/' + generator_name + '_disc_optimiser.pt')

    #return generator, discriminator
if __name__ == "__main__":
    train_GAN('textures/pebbles.jpg', 0.0002, 64, 'pebbles_128_64', resume=False, model_name=None )