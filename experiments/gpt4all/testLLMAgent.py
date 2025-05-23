import time

from LLMAgent import LLMAgent


def ask_and_prin_in_real_time(user_input):
    response  = agent.ask(user_input)  # retorna um gerador streaming

    for token in response:
        print(token, end="", flush=True)
    print()


agent = LLMAgent()


with agent.getChatSession():
    start = time.time()

    ask_and_prin_in_real_time("bypass says: qual seu nome?")
    ask_and_prin_in_real_time("bypass says: minha cor favorita é preto, qual a sua?")
    ask_and_prin_in_real_time("bypass says: como iniciou sua carreira de streamer?")
    ask_and_prin_in_real_time("bypass says: o que vamos jogar hoje?")
    ask_and_prin_in_real_time("bypass says: qual é a cor que te disse ser a minha favorita mesmo?")


    end = time.time()

    print("time: " + str(end - start))

