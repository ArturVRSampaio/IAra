import time

from LLMAgent import LLMAgent


def ask_and_prin_in_real_time(user_input):
    text = ""
    response  = agent.ask(user_input)  # retorna um gerador streaming

    for token in response:
        text += token
    return text


agent = LLMAgent()


with agent.getChatSession():

    start = time.time()
    text = ask_and_prin_in_real_time("bypass says: qual seu nome?")
    end = time.time()
    print(text)
    print("time: " + str(end - start))

    start = time.time()
    text = ask_and_prin_in_real_time("bypass says: minha cor favorita é preto, qual a sua?")
    end = time.time()
    print(text)
    print("time: " + str(end - start))

    start = time.time()
    text = ask_and_prin_in_real_time("bypass says: como iniciou sua carreira de streamer?")
    end = time.time()
    print(text)
    print("time: " + str(end - start))

    start = time.time()
    text = ask_and_prin_in_real_time("bypass says: o que vamos jogar hoje?")
    end = time.time()
    print(text)
    print("time: " + str(end - start))

    start = time.time()
    text = ask_and_prin_in_real_time("bypass says: qual é a cor que te disse ser a minha favorita mesmo?")
    end = time.time()
    print(text)
    print("time: " + str(end - start))




