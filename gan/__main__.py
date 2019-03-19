'''
Entry point into GAN module
'''
import argparse
import os
from config import BASE_DIRECTORY
from .generate import get_args_demo_ps_gan, get_args_demo_dc_gan, generate_image
from .train import get_args_train_ps_gan, get_args_train_dc_gan, train_gan
from .animate import generate_gif


def main():
    '''
    Parses arguments and calls respective functions
    '''
    parser = argparse.ArgumentParser(prog="Train and demonstrate GANs for texture generation.")
    subparsers = parser.add_subparsers(title="actions", help="train new GAN or demo existing GAN", dest='action')

    #Add parsers
    add_train_parser(subparsers)
    add_demo_parser(subparsers)
    add_animate_parser(subparsers)

    args = parser.parse_args()
    if args.action == 'train':
        if args.model == 'ps':
            params = get_args_train_ps_gan(args.source, args.name, image_size=args.image_size, scaling_factor=args.scaling_factor, iterations=args.iterations, batch_size=args.batch_size, checkpoint_frequency=args.checkpoint_frequency, resume_from=args.resume_path)
        elif args.model == 'dc':
            params = get_args_train_dc_gan(args.source, args.name, iterations=args.iterations, scaling_factor=args.scaling_factor, batch_size=args.batch_size, checkpoint_frequency=args.checkpoint_frequency, resume_from=args.resume_path)
        else:
            raise ValueError('Please select a valid GAN model (ps/dc)')
        train_gan(**params)

    elif args.action == 'demo':
        if args.model == 'ps':
            params = get_args_demo_ps_gan(args.source, args.checkpoint, args.image_size, args.tile)
        elif args.model == 'dc':
            params = get_args_demo_dc_gan(args.source, args.checkpoint)
        else:
            raise ValueError('Please select a valid GAN model (ps/dc)')
        image = generate_image(**params)
        image.save(os.path.join(BASE_DIRECTORY, 'output.jpg'))

    elif args.action == 'animate':
        generate_gif(args.source, gan_checkpoint=args.checkpoint, noise_type=args.noise_type, image_size=args.image_size, frames=args.frames, frame_duration=args.frame_duration, show_noise=args.show_noise, tile=args.tile)

def add_train_parser(subparsers):
    '''
    Add subparser for training gan models
    Args:
        subparsers:     An argparse subparsers group
    '''
    parser_train = subparsers.add_parser("train", help = "train a new GAN model")
    parser_train.add_argument('model', help='GAN model to use, either "ps" or "dc"')
    parser_train.add_argument('source', help='path to the source image file or source image folder')
    parser_train.add_argument('name', help='checkpoints will be saved under this name')
    parser_train.add_argument('-s', '--image_size', nargs='?', const=256, default=256, type=int, help='size of image to train on')
    parser_train.add_argument('-f', '--scaling_factor', nargs='?', const=1, default=1, type=int, help='images will be scaled up by this factor before cropping to size')
    parser_train.add_argument('-b', '--batch_size', nargs='?', const=8, default=8, type=int, help='how many images to train on concurrently')
    parser_train.add_argument('-i', '--iterations', nargs='?', const=44000, default=44000, type=int, help='the number of iterations over the training data')
    parser_train.add_argument('-c', '--checkpoint_frequency', nargs='?', const=1000, default=1000, type=int, help='the number of iterations between checkpoints - set to -1 to disable checkpoints')
    parser_train.add_argument('-r', '--resume_path', nargs='?', const=None, default=None, type=str, help='Directory of checkpoint to resume from')

def add_demo_parser(subparsers):
    '''
    Add subparser for generating images with gan models
    Args:
        subparsers:     An argparse subparsers group
    '''
    parser_demo = subparsers.add_parser ("demo", help = "generate images using existing GAN model")
    parser_demo.add_argument('model', help='GAN model to use, either "ps" or "dc"')
    parser_demo.add_argument('source', help='the name of the trained GAN to demo')
    parser_demo.add_argument('-s', '--image_size', nargs='?', const=256, default=256, type=int, help='size of the generated image (where applicable)')
    parser_demo.add_argument('-c', '--checkpoint', nargs='?', const=-1, default=-1, type=int, help='choose a specific saved checkpoint')
    parser_demo.add_argument('-t', '--tile', action='store_true', help='True if the generated image should be tileable (where applicable)')

def add_animate_parser(subparsers):
    '''
    Add subparser for generating animated GIFs with gan models
    Args:
        subparsers:     An argparse subparsers group
    '''
    parser_animate = subparsers.add_parser ("animate", help="generate animated GIFs using existing GAN model")
    parser_animate.add_argument('source', help='the name of the trained GAN to use')
    parser_animate.add_argument('-c', '--checkpoint', nargs='?', const=-1, default=-1, type=int, help='choose a specific saved checkpoint')
    parser_animate.add_argument('-n', '--noise_type', nargs='?', const='sin', default='sin', type=str, help='the method to generate the noise, either "sin" or "interpolated"')
    parser_animate.add_argument('-s', '--image_size', nargs='?', const=256, default=256, type=int, help='size of the generated GIF')
    parser_animate.add_argument('-f', '--frames', nargs='?', const=150, default=150, type=int, help='number of frames in the generated GIF')
    parser_animate.add_argument('-d', '--frame_duration', nargs='?', const=50, default=50, type=int, help='duration in ms of each frame in the GIF')
    parser_animate.add_argument('-v', '--show_noise', action='store_true', help='whether a graphical representation of the noise vector should be included in the GIF')
    parser_animate.add_argument('-t', '--tile', action='store_true', help='wether the generated GIF should be tileable')


if __name__ == '__main__':
    main()
