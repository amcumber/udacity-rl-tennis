"""Plotting Tools"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_scores(scores, i_map=0, count=100, solved=0.5):
    sns.set_style('darkgrid')
    sns.set_context('talk')
    sns.set_palette('Paired')
    cmap = sns.color_palette('Paired')

    scores = np.array(scores).squeeze()
    score_df = pd.DataFrame({'Scores': scores})
    score_df = score_df.assign(
        Mean=lambda df: df.rolling(count).mean()['Scores'],
        Solved=solved,
    )

    fig ,ax = plt.subplots(1,1, figsize=(10,8))

    ax = score_df[['Scores', 'Mean']].plot(
        ax=ax, color=cmap[2*(i_map%4):], alpha=0.8
    )
    ax = score_df[['Solved']].plot(ax=ax, color='red')

    ax.set_title('DDPG Scores vs Time for Tennis')
    ax.set_xlabel('Episode #')
    ax.set_ylabel('Score')
    plt.show()
    return fig
