import json
import numpy as np

from unityagents import UnityEnvironment



def main():

    config = json.load(open('config.json'))

    if config['TODO']['train']:
        print('This part is for training ...')

    if config['TODO']['test']:
        print('This part is for testing ...')

    if config['TODO']['someTest']['TODO']:
        from tests import allTests
        allTests.allTests()

    return

if __name__ == '__main__':
    main()

