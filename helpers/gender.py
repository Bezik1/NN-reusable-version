def get_gender(probability):
    if(probability >= 0.5):
      return "Male", round(probability * 100, 2)
    else:
      return "Female", round((1 - probability) * 100, 2)