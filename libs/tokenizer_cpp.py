# Custom Libs
import scribe
from .globals import HyprParams

# Training the tokenizer
tokenizer = scribe.Tokenizer()
tokenizer.train("./datasets/ri.txt", HyprParams.nVocab)
