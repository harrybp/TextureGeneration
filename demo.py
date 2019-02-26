from tkinter import *
import torch
from tkinter.ttk import *
from PIL import Image, ImageTk
import gan
import models
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Demo_Application():
    needs_update = False
    noise_vector = None
    noise_image = None
    input_image = None
    result_image = None
    generator = None

    def __init__(self, window):
        self.generator = models.PSGenerator()
        self.window = window
        window.title("GAN DEMO")

        #Column 0
        self.v = StringVar()
        self.v.trace('w',self.image_select)
        self.combo = Combobox(textvar=self.v, values=['kilburn','kitchen','tree','painting', 'water', 'snake', 'lava', 'bricks', 'pebbles', 'check', 'camo'])
        self.combo.grid(column=0, row=0)
        self.canvas1 = Canvas(width=256, height=256)
        self.canvas1.grid(column=0, row=1)
        self.canvas1.create_rectangle(0, 0, 256, 256, fill="blue")

        #Column 1
        self.buttons = Frame()
        self.buttons.grid(column=1, row=0)
        self.button1 = Button(self.buttons, text='Random Noise', command=self.generate_noise)
        self.button1.grid(column=0, row=0)
        self.fill_entry = Entry(self.buttons, width=4)
        self.fill_entry.grid(column=1, row=0)
        self.button3 = Button(self.buttons, text='Fill Vector', command=self.generate_noise_filled)
        self.button3.grid(column=2, row=0)
        self.canvas2 = Canvas(width=256, height=256)
        self.canvas2.grid(column=1, row=1)
        self.canvas2.create_rectangle(0, 0, 256, 256, fill="blue")

        #Column 2
        self.button2 = Button(text='Generate Image', command=self.gen_image)
        self.button2.grid(column=2, row=0)
        self.canvas3 = Canvas(width=256, height=256)
        self.canvas3.grid(column=2, row=1)
        self.canvas3.create_rectangle(0, 0, 256, 256, fill="blue")
        self.id = window.after(100,self.update_images)

    def update_noise_image(self):
        transform = transforms.ToPILImage()
        noise = self.noise_vector.view((4,32,32))
        img = transform(noise)
        img = img.resize((256,256), Image.NEAREST)
        self.noise_image = ImageTk.PhotoImage(img)
        self.needs_update = True

    def generate_noise_filled(self):
        fill_number = float(self.fill_entry.get())
        self.noise_vector = torch.Tensor(np.full((1, 64, int(256/32), int(256/32)),fill_number))
        self.update_noise_image()

    def generate_noise(self):
        generator = models.PSGenerator()
        self.noise_vector = generator.noise(1, 256)
        self.update_noise_image()
        
    

    def gen_image(self):
        image = self.combo.get()
        self.generator.load_state_dict(torch.load('models/'+ image +'/ps/336/generator.pt'))
        gan.demonstrate_gan(self.generator, 'demo.jpg', self.noise_vector)
        img = Image.open('demo.jpg')
        self.result_image = ImageTk.PhotoImage(img)
        self.needs_update = True

    def image_select(self, x, y, z):
        image = self.combo.get()
        img = "textures/cropped/" + image + '.jpg'
        card_PIL = Image.open(img)
        self.input_image = ImageTk.PhotoImage(card_PIL)
        self.needs_update = True

    def update_images(self):
        if self.noise_vector is None or self.input_image is None:
            self.button2['state'] = 'disabled'
        else:
            self.button2['state'] = 'normal'
        if self.needs_update:
            if self.input_image is not None:
                self.canvas1.create_image(128, 128, image=self.input_image)
            if self.noise_image is not None:
                self.canvas2.create_rectangle(0, 0, 256, 256, fill="black")
                self.canvas2.create_image(128, 128, image=self.noise_image)
            if self.result_image is not None:
                self.canvas3.create_image(128, 128, image=self.result_image)     
        self.needs_update = False
        id = self.window.after(100, self.update_images)
        

root = Tk()
Demo_Application(root)
root.mainloop()