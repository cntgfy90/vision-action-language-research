from sklearn.linear_model import LinearRegression


def measure_information_preservation(tokenizer, actions):
    tokens = tokenizer.tokenize(actions)
    reconstructed = tokenizer.detokenize(tokens)

    model = LinearRegression()
    # learn the mapping from reconstructed actions back to original actions
    model.fit(reconstructed, actions)

    # how well reconstructed predicts original
    score = model.score(reconstructed, actions)
    return score
