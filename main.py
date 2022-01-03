import argparse
import sys

from torch import load, save
from data import mnist
from model import MyAwesomeModel, fit, val


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
        save(model.state_dict(), 'model.pth')

        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        weights = load(args.load_model_from)
        model = MyAwesomeModel()
        model.load_state_dict(weights)

        _,test_set = mnist()
        accuracy = val(model, test_set)
        print(f'Accuracy: {accuracy*100}%')

 
if __name__ == '__main__':
    TrainOREvaluate()