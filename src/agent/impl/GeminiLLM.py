from src.agent import LLM

class GeminiLLM(LLM):

    def generate(self, messages, tools=None):
        # call Gemini
        # return either:
        # - normal message
        # - tool call
        pass