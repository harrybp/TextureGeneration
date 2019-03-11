'''
Entry point into GAN module
'''
import argparse
from .generate import demo_ps_gan, demo_dc_gan
from .train import train_dc_gan, train_ps_gan
from .animate import generate_gif

def main():
    '''
    Parses arguments and calls respective functions
    '''
    parser = argparse.ArgumentParser(prog="Train and demonstrate GANs for texture generation.")
    subparsers = parser.add_subparsers(title="actions", help="train new GAN or demo existing GAN", dest='action')

    #Train a new GAN
    parser_train = subparsers.add_parser("train", help = "train a new GAN model")
    parser_train.add_argument('model',  help='GAN model to use, either "ps" or "dc"')
    parser_train.add_argument('source',  help='path to the source image')
    parser_train.add_argument('name',  help='checkpoints will be saved under this name')
    parser_train.add_argument('--image_size', nargs='?', const=256, default=256, type=int,   help='size of image to train on')
    parser_train.add_argument('--batch_size', nargs='?', const=8, default=8, type=int, help='how many images to train on concurrently')
    parser_train.add_argument('--iterations', nargs='?' , const=44000, default=44000, type=int, help='the number of iterations over the training data')
    #A
    
    #Demo GAN
    parser_update = subparsers.add_parser ("demo", help = "generate images using existing GAN model")
    parser_update.add_argument('model',  help='GAN model to use, either "ps" or "dc"')
    parser_update.add_argument('source',  help='the name of the trained GAN to demo')
    parser_update.add_argument('--image_size', nargs='?', const=256, default=256, type=int, help='size of the generated image (where applicable)')
    parser_update.add_argument('--checkpoint', nargs='?', const=-1, default=-1, type=int, help='choose a specific saved checkpoint')
    parser_update.add_argument('--tile', action='store_true', help='True if the generated image should be tileable (where applicable)')

    #Create GIF
    parser_update = subparsers.add_parser ("animate", help="generate animated GIFs using existing GAN model")
    parser_update.add_argument('source', help='the name of the trained GAN to use')
    parser_update.add_argument('--checkpoint', nargs='?', const=-1, default=-1, type=int, help='choose a specific saved checkpoint')
    parser_update.add_argument('--noise_type', nargs='?', const='sin', default='sin', type=str, help='the method to generate the noise, either "sin" or "interpolated"')
    parser_update.add_argument('--image_size', nargs='?', const=256, default=256, type=int, help='size of the generated GIF')
    parser_update.add_argument('--frames', nargs='?', const=150, default=150, type=int, help='number of frames in the generated GIF')
    parser_update.add_argument('--frame_duration', nargs='?', const=50, default=50, type=int, help='duration in ms of each frame in the GIF')
    parser_update.add_argument('--show_noise', action='store_true', help='whether a graphical representation of the noise vector should be included in the GIF')
    parser_update.add_argument('--tile', action='store_true', help='wether the generated GIF should be tileable')

    args = parser.parse_args()
    if args.action == 'train':
        print('Model %s, Src %s, Name %s, size %d, batch %d, iters %d' % (args.model, args.source, args.name, args.image_size, args.batch_size, args.iterations))
        if args.model == 'ps':
            train_ps_gan(args.source, args.name, args.image_size, args.iterations, args.batch_size)
        elif args.model == 'dc':
            train_dc_gan(args.source, args.name, args.iterations, args.batch_size)
        else:
            print('Please select a valid GAN model (ps/dc)')
    elif args.action == 'demo':
        if args.model == 'ps':
            demo_ps_gan(args.source, args.checkpoint, args.image_size, args.tile)
        elif args.model == 'dc':
            demo_dc_gan(args.source, args.checkpoint)
        else:
            print('Please select a valid GAN model (ps/dc)')
    elif args.action == 'animate':
        generate_gif(args.source, gan_checkpoint=args.checkpoint, noise_type=args.noise_type, image_size=args.image_size, frames=args.frames, frame_duration=args.frame_duration, show_noise=args.show_noise, tile=args.tile)

if __name__ == '__main__':
    main()
