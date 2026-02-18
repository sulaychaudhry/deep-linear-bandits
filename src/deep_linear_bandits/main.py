from deep_linear_bandits.data import load_kuairec_big

def main():
    load_kuairec_big('/home/sulay/deep-linear-bandits/kuairec/data/big_matrix.csv')

    # read matrix
    # use watch ratio to filter (& remove watch ratio)
    # drop duplicates

    # sklearn: isolate users that I can split on
    # 80-20 split on those users
    # 80 joins the ones that I couldn't split on