import json

def main():

    config = json.load(open('config.json'))

    if config['TODO']['train']:
        from utils import trainer
        trainer.train()

    if config['TODO']['trainMemory']:
        from utils import trainerMemory
        trainerMemory.train()

    if config['TODO']['test']:
        from utils import tester
        tester.testing( **config['testing'] )

    if config['TODO']['someTest']['TODO']:
        from tests import allTests
        allTests.allTests()

    return

if __name__ == '__main__':
    main()

