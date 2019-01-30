import argparse
'''
parser = argparse.ArgumentParser(description='Train and demonstrate DCGANs for texture generation.')
parser.add_argument('--task', choices=['train','demo'],  help='either train new GAN or demo existing')

parser.add_argument('source',  help='source image to train on')



parser.add_argument('source',  help='the source image for texture style')
parser.add_argument('target',  help='the filename for the created image')
parser.add_argument('--lr', nargs='?', const=0.8, default=0.8, type=float, help='the learning rate for the optimiser')
parser.add_argument('--iter', nargs='?' , const=400, default=400, type=int, help='the number of iterations')
parser.add_argument('--tile', nargs='?' , const=False, default=False, type=bool, help='make the resulting texture tileable')

args = parser.parse_args()
print(args)
'''
parser = argparse.ArgumentParser(prog="Train and demonstrate DCGANs for texture generation.")
subparsers = parser.add_subparsers(title="actions", help="train new GAN, resume training GAN or demo GAN", dest='action')

#Train new GAN
parser_train = subparsers.add_parser("train", help = "train new")
parser_train.add_argument('source',  help='the source image to train on')
parser_train.add_argument('target',  help='the name to save the trained generator as')
parser_train.add_argument('--lr', nargs='?', const=0.0002, default=0.0002, type=float, help='the learning rate for the optimiser')
parser_train.add_argument('--iter', nargs='?' , const=64, default=64, type=int, help='the number of iterations over the training data')

#Resume training GAN
parser_resume = subparsers.add_parser ("resume", help = "resume training")
parser_resume.add_argument('source',  help='the source image to train on')
parser_resume.add_argument('target',  help='the name to save the trained generator as')
parser_resume.add_argument('source_GAN',  help='the name of the saved GAN you want to resume training')
parser_resume.add_argument('--lr', nargs='?', const=0.0002, default=0.0002, type=float, help='the learning rate for the optimiser')
parser_resume.add_argument('--iter', nargs='?' , const=64, default=64, type=int, help='the number of iterations over the training data')

#Demo GAN
parser_update = subparsers.add_parser ("demo", help = "demo existing")
parser_update.add_argument('source',  help='the name of the trained GAN to demo')
args = parser.parse_args()
'''
if args.action == 'train':
    train_GAN(args.source, args.lr, args.iter, args.target)
elif args.action == 'resume':
    train_GAN(args.source, args.lr, args.iter, args.target, resume=True, model_name=args.source_GAN)
elif args.action == 'demo':
    demo_GAN(args.source)
'''
import os
if not os.path.exists('models2'):
    os.makedirs('models2')

print(args)
#print(args2)
#def train_GAN(source_image, learning_rate, iterations, generator_name, resume=False, model_name=None ):