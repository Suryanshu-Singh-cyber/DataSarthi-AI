def recommend_model(rows, cols):
    if rows < 1000:
        return "Naive Bayes"
    elif cols > 10:
        return "SVM"
    else:
        return "Random Forest"
