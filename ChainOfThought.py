import dspy

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer") # add relation which columns are related

    def forward(self, question):
        return self.prog(question=question)
    