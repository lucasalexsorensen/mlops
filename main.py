import argparse
import sys

import torch

from data import mnist
from model import MyAwesomeModel, fit


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        train_set, _ = mnist()

        model = MyAwesomeModel()
        fit(parser, model, train_set)
        torch.save(model.state_dict(), 'model.pth')

        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        weights = torch.load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(weights)

        model.eval()
        _, test_set = mnist()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_set:
                images = images.view(images.shape[0], -1)
                ps = model(images)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                total += images.shape[0]
                correct += torch.sum(equals)
        accuracy = float(correct) / float(total)
        print(f'Accuracy: {accuracy*100}%')

 
if __name__ == '__main__':
    TrainOREvaluate()