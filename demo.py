from tkinter import *
import torch
from tkinter.ttk import *
from PIL import Image, ImageTk
import gan
import models
import numpy as np
import torchvision.transforms as transforms
import torchvision.utils as vutils

class Demo():
    needs_update = False
    noise_vector = None
    noise_image = None
    input_image = None
    result_image = None
    generator = None

    def __init__(self, window):
        self.window = window
        
        self.select_container = Frame()
        self.sel = StringVar()
        self.sel.trace('w',self.update_generator)
        self.select_gan = Combobox(self.select_container, textvar=self.sel, values=['ps','dc'])
        self.select_gan.set('ps')
        self.select_gan.pack(side='right')
        self.select_label = Label(self.select_container, text='GAN model ')
        self.select_label.pack(side='left')
        self.select_container.grid(column=1, row=0)

        self.space = Label()
        self.space.grid(column=1, row=1)

        #Column 0
        self.img_select_container = Frame()
        self.v = StringVar()
        self.v.trace('w',self.image_select)
        self.combo = Combobox(self.img_select_container, textvar=self.v, values=['kilburn', 'tree','painting', 'water', 'snake', 'lava', 'bricks', 'pebbles', 'check', 'camo', 'paint_wood'])
        self.combo.pack(side='right')
        self.img_select_label = Label(self.img_select_container, text='Source Image ')
        self.img_select_label.pack(side='left')
        self.img_select_container.grid(column=0, row=2)


        self.canvas1 = Canvas(width=256, height=256)
        self.canvas1.grid(column=0, row=3)
        self.canvas1.create_rectangle(0, 0, 256, 256, fill="blue")

        #Column 1
        self.buttons = Frame()
        self.buttons.grid(column=1, row=2)
        self.button1 = Button(self.buttons, text='Random Noise', command=self.generate_noise)
        self.button1.grid(column=0, row=2)
        self.fill_entry = Entry(self.buttons, width=4)
        self.fill_entry.grid(column=1, row=2)
        self.button3 = Button(self.buttons, text='Fill Vector', command=self.generate_noise_filled)
        self.button3.grid(column=2, row=2)
        self.canvas2 = Canvas(width=256, height=256)
        self.canvas2.grid(column=1, row=3)
        self.canvas2.create_rectangle(0, 0, 256, 256, fill="blue")

        #Column 2
        self.button2 = Button(text='Generate Image', command=self.gen_image)
        self.button2.grid(column=2, row=2)
        self.canvas3 = Canvas(width=256, height=256)
        self.canvas3.grid(column=2, row=3)
        self.canvas3.create_rectangle(0, 0, 256, 256, fill="blue")
        self.id = window.after(100,self.update_images)
        self.update_generator(1,2,3)

    def update_generator(self, x, y, z):
        if self.select_gan.get() == 'ps':
            self.generator = models.PSGenerator()
        elif self.select_gan.get() == 'dc':
            self.generator = models.DCGenerator(100,64,3)

    def update_noise_image(self):
        transform = transforms.ToPILImage()
        if self.select_gan.get() == 'ps':
            noise = self.noise_vector.view((4,32,32))
        elif self.select_gan.get() == 'dc':
            noise = self.noise_vector.view((1,10,10))
        img = transform(noise)
        img = img.resize((256,256), Image.NEAREST)
        self.noise_image = ImageTk.PhotoImage(img)
        self.needs_update = True

    def generate_noise_filled(self):
        fill_number = float(self.fill_entry.get())
        if self.select_gan.get() == 'ps':
            noise = torch.Tensor(np.random.uniform(-1, 1, (1, 64, int(256/64), int(256/32)))) #Generate batch of noise for input
            filled = torch.Tensor(np.full((1, 64, int(256/64), int(256/32)),fill_number))
            self.noise_vector = torch.cat((noise,filled),2)
        elif self.select_gan.get() == 'dc':
            self.noise_vector = torch.Tensor(np.full((1, 100),fill_number))
        self.update_noise_image()

    def generate_noise(self):
        self.noise_vector = self.generator.noise(1, 256)
        self.update_noise_image()
        
    def gen_image(self):
        image = self.combo.get()
        
        self.generator.load_state_dict(torch.load('models/'+ image +'/'+ self.select_gan.get() +'/336/generator.pt'))
        if image=='kilburn':
            self.generator.load_state_dict(torch.load('models/'+ image +'/'+ self.select_gan.get() +'/453/generator.pt'))
        gan.generate_image(self.generator, 'demo.jpg', self.noise_vector)
        img = Image.open('demo.jpg')
        img = img.resize((256,256), Image.NEAREST)
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
root.title("GAN DEMO")
frame = Frame(root)
frame.grid()
Demo(frame)
root.mainloop()