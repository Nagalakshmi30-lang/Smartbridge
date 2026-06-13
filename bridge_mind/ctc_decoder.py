import torch

class CTCDecoder:
    def __init__(self, labels):
        self.labels = labels

    def greedy(self, logits):
        probs = torch.argmax(logits, dim=-1)
        prev = None
        output = []

        for p in probs:
            if p != prev and p != 0:
                output.append(self.labels[p])
            prev = p

        return " ".join(output)
