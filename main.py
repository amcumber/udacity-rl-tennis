"""Main Script for tennis_agents"""
from tennis_agents.factory import TennisFactory
from tennis_agents import plotting

def main(file:str) -> None:
    """
    Main Function that will load configuration from file, train, and plot the
    resulting scores
    """
    factory = TennisFactory()
    trainer = factory(file)

    scores = trainer.train()
    plotting.plot_scores(scores)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Run Tennis from terminal')
    p.add_argument('config', help='Configuration File - toml')
    args = p.parse_args()

    main(args.config)
