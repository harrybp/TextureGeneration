import argparse
from .generate import generate_texture

def main():
    parser = argparse.ArgumentParser(description='Generate texture using gatys et al method.')
    parser.add_argument('source',  help='the source image for texture style')
    parser.add_argument('-l', '--learning_rate', nargs='?', const=0.8, default=0.8, type=float, help='the learning rate for the optimiser')
    parser.add_argument('-s', '--image_size', nargs='?' , const=128, default=128, type=int, help='the size of the generated image')
    parser.add_argument('-i', '--iterations', nargs='?' , const=150, default=150, type=int, help='the number of iterations')
    parser.add_argument('-t', '--tile', action='store_true', help='make the resulting texture tileable')
    args = parser.parse_args()
    generate_texture(args.source, args.learning_rate, args.iterations, image_size=args.image_size, tile=args.tile)

if __name__ == "__main__":
    main()