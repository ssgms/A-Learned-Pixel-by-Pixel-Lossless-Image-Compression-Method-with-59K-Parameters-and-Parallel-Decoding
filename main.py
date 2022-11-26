import argparse
from config import *
from agents import *

def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument('config',metavar='config',default='None',help='The Configuration file in json format')
    args = arg_parser.parse_args()
    config, _ = get_config_from_json(args.config)
    config = process_config(config)
    run_agent(config)

def run_agent(config):
    agent_class = globals()["MaskedConvLosslessAgent"]
    agent = agent_class(config)
    agent.run()
    agent.finalize()

if __name__ == '__main__':
    main() 
    
